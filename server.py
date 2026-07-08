import sys
import os

# 必须在导入任何模型前设置，避免 subprocess DEVNULL 时 tqdm 崩溃
os.environ.setdefault("TQDM_DISABLE", "1")

# Windows 终端默认 GBK，工具/API 返回含拉丁字母等字符时 print 会抛 UnicodeEncodeError 导致终端无输出
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 全局 safe print：subprocess DEVNULL 时 stdout 为关闭的文件，任何 print 都会抛异常
_builtin_print = print
def _safe_print(*args, **kwargs):
    try:
        _builtin_print(*args, **kwargs)
    except Exception:
        pass
import builtins
builtins.print = _safe_print

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, HTTPException
from starlette.responses import FileResponse
from langchain_openai import ChatOpenAI
import uvicorn
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import requests
import pandas as pd
import difflib
import re
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
import asyncio
import uuid
import edge_tts
from langchain_community.utilities import SerpAPIWrapper
from odds_api import OddsAPIClient
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "")
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
odds_API_KEY = os.getenv("ODDS_API_KEY", "your_odds_api_key")
odds_client = OddsAPIClient(api_key=odds_API_KEY)

app = FastAPI()

from lightrag_kb import (
    _ensure_init,
    get_rag,
    insert_knowledge_sync,
    search_knowledge_sync,
)


@app.on_event("startup")
async def startup_lightrag():
    """fastapi 启动时自动初始化 LightRAG 知识库"""
    await _ensure_init()


def _print_tool_preview(tag: str, payload, max_chars: int = 1200) -> None:
    """在终端打印工具真实返回（截断版），便于排查模型幻觉/空回答。"""
    try:
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        text = str(payload)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (截断)"
    try:
        print(f"[{tag}] 查询结果:\n{text}")
    except (UnicodeEncodeError, ValueError, OSError):
        pass


def _strip_halfwidth_parens_for_tts(s: str) -> str:
    """去掉半角 ( ) 内内容；保留明显为「比分」的短数字段，如 (1-1)、(2:1)。"""
    rx = re.compile(r"\([^)]*\)")
    score_inner = re.compile(r"^\s*\d{1,2}\s*[-:：]\s*\d{1,2}\s*$")
    spans = [
        (m.start(), m.end())
        for m in rx.finditer(s)
        if not score_inner.fullmatch(s[m.start() + 1 : m.end() - 1].strip())
    ]
    for a, b in reversed(spans):
        s = s[:a] + s[b:]
    return s


def _speech_text_for_tts(text: str) -> str:
    """从完整回复里抽出适合朗读的对白：去掉全角（）动作，以及半角 () 里的动作/颜文字（保留类似 1-1 的比分）。"""
    if not text:
        return ""
    s = text
    # 全角（）动作/神态
    fw = re.compile(r"（[^）]*）")
    while True:
        n = fw.sub("", s)
        if n == s:
            break
        s = n
    # 半角 ()：如 (轻轻撩了下头发)、(*^▽^*)；保留 (1-1) 类比分
    s = _strip_halfwidth_parens_for_tts(s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = "\n".join(lines)
    s = re.sub(r"[ \t\u3000]{2,}", " ", s)
    return s.strip()


def _strip_agent_output_leak(text: str) -> str:
    """去掉模型偶发泄漏到正文里的思考标签、伪工具 XML（不影响 TTS 专用规则）。"""
    if not text:
        return ""
    s = text
    patterns = [
        re.compile(r"<redacted[^>]+>[\s\S]*?</\s*redacted[^>]+>", re.I | re.DOTALL),
        re.compile(r"</?redacted[^>]{0,64}>", re.I),
        re.compile(r"<tool_call[\s\S]*?</tool_call>", re.I),
        re.compile(r"</?tool_call[^>]{0,64}>", re.I),
        re.compile(r"<function[^>]*>[\s\S]*?</function>", re.I),
        re.compile(r"</?function[^>]{0,64}>", re.I),
        re.compile(r"<parameter[^>]*>[\s\S]*?</parameter>", re.I),
        re.compile(r"</?parameter[^>]{0,64}>", re.I),
        re.compile(r"<\s*think\s*>[\s\S]*?</\s*think\s*>", re.I),
    ]
    for _ in range(4):
        prev = s
        for p in patterns:
            s = p.sub("", s)
        if s == prev:
            break
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _openai_messages_to_question(messages: list) -> str:
    """把 OpenAI 风格的 messages 拼成一段文本，交给本地 ask_agent。"""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        content = m.get("content")
        if isinstance(content, list):
            # 多模态片段：只拼接 text 类型
            texts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(t for t in texts if t)
        if not isinstance(content, str) or not content.strip():
            continue
        if role in ("system", "user", "assistant", "tool"):
            parts.append(f"[{role}]\n{content.strip()}")
    return "\n\n".join(parts)


# memory buffer
class SimpleSummaryBuffer:
    def __init__(self, llm, max_token_limit=100):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.messages = []
        self.buffer = ""

    def load_memory_variables(self, _):
        if self.buffer:
            msgs = [SystemMessage(content=f"[摘要] {self.buffer}")]
        else:
            msgs = []
        msgs.extend(self.messages)
        return {'history': msgs}

    def save_context(self, inputs, outputs):
        self.messages.append(HumanMessage(content=inputs['input']))
        self.messages.append(AIMessage(content=outputs))
        # 超长时生成摘要
        total_tokens = sum(len(m.content) for m in self.messages) // 2
        while total_tokens > self.max_token_limit and len(self.messages) > 1:
            old = self.messages[:-1]
            self.messages = self.messages[-1:]
            summary = self.llm.invoke('用一句话总结：' + '\n'.join(m.content for m in old))
            self.buffer = summary.content if hasattr(summary, 'content') else str(summary)


@tool
def multiply(a: int, b: int) -> int:
    '''计算两个整数的乘积'''
    # print(f'agent使用了multiply工具, 计算{a}*{b}的结果')
    return a * b


city_cn = pd.read_csv('city_cn.csv', sep='\t')
city_in = pd.read_csv('city_in.csv', sep='\t')

def get_city_code(city_name: str) -> str:
    '''
    获取城市编码
    '''
    # 匹配国内城市
    # 优先匹配区县
    match = city_cn[city_cn['district'] == city_name]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']
    
    # 匹配城市
    match = city_cn[city_cn['city'].str.contains(city_name, case=False)]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']
    
    # 匹配省份
    match = city_cn[city_cn['province'].str.contains(city_name, case=False)]
    if not match.empty:
        return match.iloc[0]['areacode/城市ID']

    # 匹配国际城市
    match = city_in[city_in['level3_chn'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme'] 
    
    match = city_in[city_in['level2_chn'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme']

    match = city_in[city_in['level1_chn'].str.contains(city_name, case=False)]
    if not match.empty:
        return match.iloc[0]['meme']  

    match = city_in[city_in['country_chn'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme']

    match = city_in[city_in['level3_eng'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme']

    match = city_in[city_in['level2_eng'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme'] 

    match = city_in[city_in['level1_eng'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme']
    
    match = city_in[city_in['country_eng'] == city_name]
    if not match.empty:
        return match.iloc[0]['meme']

    # 默认北京
    return '101010100'

@tool
def get_weather(city: str) -> str:
    '''
    调用实时天气API, 返回温度及天气状况
    参数:
        city: 城市名称，例如：'北京'
    '''
    url = "https://eolink.o.apispace.com/456456/weather/v001/now"

    city_code = get_city_code(city)

    payload = {"areacode" : city_code}

    headers = {
        "X-APISpace-Token":"your_weather_api_token"
    }

    response=requests.request("GET", url, params=payload, headers=headers)

    # 将结果转换成 JSON 数据
    data = response.json()
    temp = data['result']['realtime']['temp']
    wd = data['result']['realtime']['text']

    if city_code == '101010100':
        return '未找到城市编码, 默认返回北京。北京的天气状况是{wd}, 温度是{temp}℃'

    return f'{city}的天气状况是{wd}, 温度是{temp}℃'

@tool
def search_web(query: str) -> str:
    '''
    只有需要了解实其他工具不能了解的时信息或者不知道的事情的时候才使用这个工具。
    使用前需询问用户是否允许，否则不能使用。
    '''
    serp = SerpAPIWrapper()
    result = serp.run(query)
    _print_tool_preview("search_web", result)
    return result

@tool
def search_match_schedule(query: str) -> str:
    """
    当 get_upcoming_events 无法获取到赛事信息时（如世界杯、亚洲杯、欧洲杯等大型国家队杯赛），
    使用此工具通过搜索引擎查找对阵、赛程等基本赛事信息。
    此工具仅用于获取赛程/对阵/时间等基础信息，不包含赔率数据。
    Args:
        query: 搜索查询，请包含赛事名称和日期，如 "2026 World Cup matches today", "世界杯决赛对阵"
    Returns:
        搜索结果摘要。
    """
    serp = SerpAPIWrapper()
    result = serp.run(query + " football match schedule fixtures")
    _print_tool_preview("search_match_schedule", result)
    return result

@tool
def get_team_details(team_name: str) -> str:
    """
    获取指定球队的详细赛前信息：主教练及教练组、关键球员与阵容、近期比赛战绩、球队统计数据。
    在进行赛前分析时，先调用此工具获取双方球队信息。
    注意：此工具返回的是搜索片段，不含赔率数据，赔率请用 get_event_odds。
    Args:
        team_name: 球队名称（国家名/俱乐部名），如 "Argentina", "Cabo Verde", "France", "Manchester City"
    Returns:
        结构化文本（JSON），包含教练信息、统计数据、阵容/关键球员、近期战绩。
    """
    serp = SerpAPIWrapper()
    results = {}

    try:
        coach_result = serp.run(f"{team_name} national football team head coach manager 2026")
        results["coach"] = str(coach_result)
    except Exception as e:
        results["coach"] = f"获取失败: {e}"

    try:
        stats_result = serp.run(f"{team_name} national football team recent match statistics goals scored conceded 2026")
        results["stats"] = str(stats_result)
    except Exception as e:
        results["stats"] = f"获取失败: {e}"

    try:
        squad_result = serp.run(f"{team_name} national team squad key players roster lineup 2026 World Cup")
        results["squad"] = str(squad_result)
    except Exception as e:
        results["squad"] = f"获取失败: {e}"

    output = json.dumps(results, ensure_ascii=False, indent=2)
    _print_tool_preview("get_team_details", output)
    return output

@tool
def search_team_injury(team_name: str) -> str:
    """
    搜索指定球队的伤病、停赛、预计首发阵容信息。
    在赛前分析时调用此工具，了解双方球队的人员可用情况，避免预测时忽略关键球员缺阵。
    Args:
        team_name: 球队名称，如 "Argentina", "Cabo Verde"
    Returns:
        伤病/停赛/首发相关信息的搜索结果。
    """
    serp = SerpAPIWrapper()
    results = {}

    try:
        injury_result = serp.run(f"{team_name} injury suspension news 2026 World Cup")
        results["injury_suspension"] = str(injury_result)
    except Exception as e:
        results["injury_suspension"] = f"获取失败: {e}"

    try:
        lineup_result = serp.run(f"{team_name} predicted starting lineup formation 2026 World Cup")
        results["predicted_lineup"] = str(lineup_result)
    except Exception as e:
        results["predicted_lineup"] = f"获取失败: {e}"

    output = json.dumps(results, ensure_ascii=False, indent=2)
    _print_tool_preview("search_team_injury", output)
    return output

@tool
def store_team_reports(team_name: str, num_matches: int = 5) -> str:
    """
    搜索指定球队的近期比赛报告，并将结果存入 LightRAG 知识库供后续检索分析。
    在赛前分析时，先调用此工具自动收集双方球队的比赛数据入库，再用 search_match_knowledge 检索。
    Args:
        team_name: 球队名称（国家名/俱乐部名），如 "Argentina", "Cabo Verde"
        num_matches: 要收集的最近比赛场次数，默认5
    Returns:
        入库结果摘要，包含存储了多少条比赛信息。
    """
    serp = SerpAPIWrapper()
    stored = 0
    reports = []

    queries = [
        f"{team_name} recent matches results scores 2026 World Cup",
        f"{team_name} match report match recap world cup 2026",
    ]

    for q in queries:
        try:
            result = serp.run(q + " football")
            text = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            reports.append(text)
        except Exception as e:
            reports.append(f"搜索失败: {e}")

    combined = f"[球队: {team_name}]\n近{num_matches}场比赛信息:\n" + "\n---\n".join(reports)

    if insert_knowledge_sync(combined):
        stored += 1

    _print_tool_preview("store_team_reports", {"team": team_name, "stored": stored})
    return json.dumps({"status": "ok", "team": team_name, "reports_stored": stored,
                       "summary": f"已将 {team_name} 的比赛信息存入知识库，可用 search_match_knowledge 检索"},

                      ensure_ascii=False)

@tool
def search_match_knowledge(query: str) -> str:
    """
    从 LightRAG 知识库中检索已存储的赛后报告和球队信息。
    必须是先通过 store_team_reports 存入数据后才能使用。
    Args:
        query: 检索查询，如 "阿根廷最近比赛的表现", "佛得角防守阵型",
               "Argentina vs Jordan match result"
    Returns:
        从知识库检索到的相关内容。
    """
    result = search_knowledge_sync(query)
    _print_tool_preview("search_match_knowledge", result)
    return result

@tool
def get_upcoming_events(sport: str = "football", league: str = "international-clubs-uefa-champions-league") -> str:
    """
    获取即将开始的体育赛事列表。只有需要了解即将开始的体育赛事的时候才使用这个工具。
    Args:
        sport: 运动类型，如 'football', 'basketball', 'tennis'
        league: 联赛名称，如 'international-clubs-uefa-champions-league', 'england-premier-league'
    Returns:
        包含赛事信息的字符串。
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            events = odds_client.get_events(sport=sport, league=league)
            break
        except requests.exceptions.ConnectTimeout as e:
            if attempt == max_retries:
                print(f"获取赛事失败(连接超时): {str(e)}")
                return (
                    "获取赛事失败：连接超时，请稍后重试，"
                    "或尝试更换联赛参数（例如 england-premier-league）。"
                )
            time.sleep(attempt * 2)
        except requests.exceptions.Timeout as e:
            if attempt == max_retries:
                print(f"获取赛事失败(请求超时): {str(e)}")
                return "获取赛事失败：请求超时，请稍后重试。"
            time.sleep(attempt * 2)
        except requests.exceptions.RequestException as e:
            # 其他网络异常直接结束，避免无效重试
            print(f"获取赛事失败(网络异常): {str(e)}")
            return f"获取赛事失败(网络异常): {str(e)}"
        except Exception as e:
            print(f"获取赛事失败: {str(e)}")
            return f"获取赛事失败: {str(e)}"

    try:
        if not events:
            print(f"未找到 {sport} - {league} 的 upcoming 赛事。")
            return (
                f"未找到 {sport} - {league} 的 upcoming 赛事（该联赛可能不在 Odds API 覆盖范围内，"
                f"如世界杯/亚洲杯/欧洲杯等国家队杯赛）。\n"
                f"请改用 search_match_schedule 工具搜索该赛事的对阵与时间信息。"
            )

        _print_tool_preview("get_upcoming_events", events)
        return json.dumps(events, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"赛事数据格式化失败: {str(e)}")
        return f"赛事数据格式化失败: {str(e)}"

@tool
def find_arbitrage_opportunities(bookmakers: str = "Bet365,Singbet") -> str:
    """
    发现不同博彩公司之间的套利机会（无风险投注）。只有需要了解即将开始的体育赛事的时候才使用这个工具。
    Args:
        bookmakers: 博彩公司名称，用逗号分隔，如 'bet365,singbet'
    Returns:
        套利机会的详细信息。
    """
    bookmaker_alias = {
        # 常见输入别名映射到更通用写法，后续还会用官方列表二次校正
        "Bet365": "bet365",
        "Singbet": "singbet",
    }

    pre_normalized = [
        bookmaker_alias.get(item.strip(), item.strip()).lower()
        for item in bookmakers.split(",")
        if item.strip()
    ]
    normalized_bookmakers = ",".join(pre_normalized)

    def _extract_valid_bookmakers(payload):
        valid = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    val = item.get("key") or item.get("name") or item.get("id")
                    if val:
                        valid.append(str(val))
                elif isinstance(item, str):
                    valid.append(item)
        elif isinstance(payload, dict):
            for key in ("data", "bookmakers", "results"):
                vals = payload.get(key)
                if isinstance(vals, list):
                    for item in vals:
                        if isinstance(item, dict):
                            val = item.get("key") or item.get("name") or item.get("id")
                            if val:
                                valid.append(str(val))
                        elif isinstance(item, str):
                            valid.append(item)
                    if valid:
                        break
        return valid

    # 请求前先拿官方 bookmaker 列表做精确校正，避免 bet365/bet-365 这类误判
    try:
        bm_resp = requests.get(
            "https://api2.odds-api.io/v3/bookmakers",
            params={"apiKey": odds_API_KEY},
            timeout=15
        )
        bm_resp.raise_for_status()
        valid_names = _extract_valid_bookmakers(bm_resp.json())
        if valid_names:
            lower_to_original = {name.lower(): name for name in valid_names}
            canonical = [lower_to_original.get(name.lower(), name) for name in pre_normalized]
            normalized_bookmakers = ",".join(canonical)
    except Exception:
        # 列表拉取失败时继续使用本地规范化结果，不阻塞主流程
        pass

    try:
        arb_bets = odds_client.get_arbitrage_bets(
            bookmakers=normalized_bookmakers,
            limit=20,
            include_event_details=True
        )
        if not arb_bets:
            print(f"未在 {normalized_bookmakers} 中找到套利机会。")
            return f"未在 {normalized_bookmakers} 中找到套利机会。"
        _print_tool_preview("find_arbitrage_opportunities", arb_bets)
        result = []
        for arb in arb_bets:
            # 根据实际返回字段调整
            event = arb.get('event', {})
            arb_percentage = arb.get('arb_percentage', 0)
            result.append(
                f"赛事: {event.get('name', '未知')}, "
                f"套利收益率: {arb_percentage:.2f}%, "
                f"投注详情: {arb.get('bets')}"
            )
        _print_tool_preview("find_arbitrage_opportunities", result)
        return "\n".join(result) + "\n注意：套利存在执行风险，请谨慎参考。"
    except Exception as e:
        err = str(e)
        if "not a valid bookmaker" in err:
            try:
                # 拉取官方可用列表，返回更友好的修复建议
                bm_resp = requests.get(
                    "https://api2.odds-api.io/v3/bookmakers",
                    params={"apiKey": odds_API_KEY},
                    timeout=15
                )
                bm_resp.raise_for_status()
                bm_data = bm_resp.json()

                valid_names = _extract_valid_bookmakers(bm_data)

                raw_inputs = [x.strip() for x in bookmakers.split(",") if x.strip()]
                suggestions = []
                if valid_names:
                    for raw in raw_inputs:
                        guess = difflib.get_close_matches(raw.lower(), [v.lower() for v in valid_names], n=3, cutoff=0.5)
                        if guess:
                            suggestions.append(f"{raw} -> {', '.join(guess)}")

                suggestion_text = "\n".join(suggestions) if suggestions else "请先调用 /v3/bookmakers 查询支持的博彩公司。"
                msg = (
                    "寻找套利机会失败：包含无效的 bookmaker。\n"
                    f"原始输入: {bookmakers}\n"
                    f"规范化后: {normalized_bookmakers}\n"
                    f"建议:\n{suggestion_text}"
                )
                print(msg)
                return msg
            except Exception:
                pass

        print(f"寻找套利机会失败: {err}")
        return f"寻找套利机会失败: {err}"

@tool
def get_event_odds(event_id: int, bookmakers: str = "Bet365") -> str:
    """
    获取指定比赛(event)的赔率信息。
    官方推荐方法：get_event_odds(event_id, bookmakers)
    Args:
        event_id: 赛事 ID，如 69339430
        bookmakers: 博彩公司，多个用逗号分隔，如 "bet365,singbet"
    Returns:
        单场赔率 JSON 字符串
    """
    try:
        requested = [item.strip() for item in bookmakers.split(",") if item.strip()]
        primary = requested[0] if requested else "Bet365"
        candidates = []
        if primary.lower() == "singbet":
            candidates.append("SingBet")
        else:
            candidates.append("Bet365")
        for bk in ("Bet365", "SingBet"):
            if bk not in candidates:
                candidates.append(bk)

        last_msg = ""
        for bk in candidates:
            try:
                odds = odds_client.get_event_odds(event_id=event_id, bookmakers=bk)
                if odds:
                    compact = {
                        "id": odds.get("id"),
                        "home": odds.get("home"),
                        "away": odds.get("away"),
                        "date": odds.get("date"),
                        "status": odds.get("status"),
                        "league": odds.get("league"),
                        "urls": odds.get("urls", {}),
                        "bookmakerIds": odds.get("bookmakerIds", {}),
                        "markets": {}
                    }
                    raw_bookmakers = odds.get("bookmakers", {})
                    markets = raw_bookmakers.get(bk, []) if isinstance(raw_bookmakers, dict) else []
                    # 仅保留关键市场，避免超大 JSON 拖慢模型生成
                    important = ["ML", "Draw No Bet", "Double Chance", "Over/Under"]
                    picked = []
                    for m in markets:
                        if isinstance(m, dict) and m.get("name") in important:
                            picked.append(m)
                        if len(picked) >= 6:
                            break
                    compact["markets"][bk] = picked

                    _print_tool_preview("get_event_odds", compact)
                    return json.dumps(compact, ensure_ascii=False, indent=2)
                last_msg = f"未获取到 event_id={event_id} 在 {bk} 的赔率。"
            except Exception as inner_e:
                last_msg = str(inner_e)

        return f"未获取到活动赔率。已尝试博彩公司: {', '.join(candidates)}。最后信息: {last_msg}"
    except Exception as e:
        return f"获取活动赔率失败: {str(e)}"

@tool
def get_bookmakers(max_items: int = 30) -> str:
    """
    获取可用的博彩公司摘要列表（默认最多返回30条，避免上下文过载）。
    Returns:
        博彩公司列表（JSON 字符串）
    """
    try:
        data = odds_client.get_bookmakers()
        total = len(data) if isinstance(data, list) else 0
        active_list = []
        inactive_list = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name")
                    active = bool(item.get("active", False))
                    if name:
                        (active_list if active else inactive_list).append(str(name))
                elif isinstance(item, str):
                    active_list.append(item)

        limit = max(1, min(int(max_items), 100))
        payload = {
            "total": total,
            "active_count": len(active_list),
            "inactive_count": len(inactive_list),
            "active_sample": active_list[:limit],
            "inactive_sample": inactive_list[: min(10, len(inactive_list))],
            "note": "如需完整列表，请明确要求“完整bookmakers列表”。"
        }
        _print_tool_preview("get_bookmakers", payload)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取可用博彩公司失败: {str(e)}"

@tool
def get_selected_bookmakers() -> str:
    """
    获取当前已选择的博彩公司。
    Returns:
        已选择博彩公司列表（JSON 字符串）
    """
    try:
        data = odds_client.get_selected_bookmakers()
        _print_tool_preview("get_selected_bookmakers", data)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取已选择博彩公司失败: {str(e)}"

@tool
def select_bookmakers(bookmakers: str = "bet365") -> str:
    """
    设置（精选）要使用的博彩公司。
    Args:
        bookmakers: 逗号分隔的博彩公司 key，如 "bet365,singbet"
    Returns:
        设置结果（JSON 或文本）
    """
    try:
        items = [b.strip().lower() for b in bookmakers.split(",") if b.strip()]
        if not items:
            return "未提供有效博彩公司参数。"
        data = odds_client.select_bookmakers(bookmakers=items)
        _print_tool_preview("select_bookmakers", data)
        return json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data
    except Exception as e:
        return f"设置精选博彩公司失败: {str(e)}"

@tool
def clear_selected_bookmakers() -> str:
    """
    清空当前已选择的博彩公司。
    Returns:
        清空结果（JSON 或文本）
    """
    try:
        data = odds_client.clear_selected_bookmakers()
        _print_tool_preview("clear_selected_bookmakers", data)
        return json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data
    except Exception as e:
        return f"清空已选择博彩公司失败: {str(e)}"

@tool
def analyze_match_for_betting(home_team: str, away_team: str, league: str = "international-clubs-uefa-champions-league") -> str:
    """
    基于即将开始的赛事列表，输出指定对阵的下注辅助信息（含时间、赛事ID、基础风险提示）。
    仅做信息与概率分析，不保证结果。
    Args:
        home_team: 主队英文名，如 "Paris Saint-Germain"
        away_team: 客队英文名，如 "Bayern Munich"
        league: 联赛 slug
    Returns:
        分析所需的结构化文本（JSON 字符串）
    """
    try:
        events = odds_client.get_events(sport="football", league=league)
        if not events:
            return f"未找到联赛 {league} 的赛事。"

        home_kw = home_team.strip().lower()
        away_kw = away_team.strip().lower()
        matched = []
        for e in events:
            home = str(e.get("home", "")).lower()
            away = str(e.get("away", "")).lower()
            if home_kw in home and away_kw in away:
                matched.append(e)

        if not matched:
            # 若严格匹配不到，给一个模糊匹配提示
            candidates = []
            for e in events[:10]:
                candidates.append(f"{e.get('home', '未知')} vs {e.get('away', '未知')} @ {e.get('date', '未知时间')}")
            return (
                f"未找到 {home_team} vs {away_team} 的精确对阵。\n"
                "可参考以下近期候选：\n" + "\n".join(candidates)
            )

        payload = {
            "query": {"home_team": home_team, "away_team": away_team, "league": league},
            "matched_events": matched[:3],
            "analysis_hint": [
                "先基于赛事时间、主客场、历史强弱给出胜平负倾向",
                "再给出2-3个可能比分（如1-1/1-2/2-1）及简短理由",
                "明确这是概率分析，不是确定结果"
            ]
        }
        _print_tool_preview("analyze_match_for_betting", payload)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"比赛分析数据获取失败: {str(e)}"

@tool
def predict_match_score_with_odds(
    home_team: str,
    away_team: str,
    bookmaker: str = "Bet365",
    league: str = "international-clubs-uefa-champions-league"
) -> str:
    """
    一步式下注辅助：自动查对阵、获取 event_id、拉取该场赔率，供后续比分预测使用。
    Args:
        home_team: 主队英文名
        away_team: 客队英文名
        bookmaker: 博彩公司（默认 bet365）
        league: 联赛 slug
    Returns:
        包含赛事基础信息与赔率信息的 JSON 字符串
    """
    try:
        events = odds_client.get_events(sport="football", league=league)
        if not events:
            return f"未找到联赛 {league} 的赛事，无法进行一步式分析。"

        home_kw = home_team.strip().lower()
        away_kw = away_team.strip().lower()
        matched_event = None
        for e in events:
            home = str(e.get("home", "")).lower()
            away = str(e.get("away", "")).lower()
            if home_kw in home and away_kw in away:
                matched_event = e
                break

        if not matched_event:
            return (
                f"未找到 {home_team} vs {away_team} 的赛事，"
                "请确认球队英文名后重试。"
            )

        event_id = matched_event.get("id")
        if not event_id:
            return "已匹配到赛事，但未获取到 event_id，无法查询赔率。"

        bookmaker_norm = bookmaker.strip() or "Bet365"
        odds_text = get_event_odds(event_id=event_id, bookmakers=bookmaker_norm)
        odds = None
        if isinstance(odds_text, str):
            try:
                odds = json.loads(odds_text)
            except Exception:
                odds = None

        payload = {
            "query": {
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker_norm,
                "league": league
            },
            "event": matched_event,
            "odds": odds if odds else None,
            "odds_fetch_message": None if odds else odds_text,
            "confidence_rule": (
                "若 odds 为空，仅可做低置信度预测；"
                "若 odds 非空，可做常规置信度预测。"
            )
        }
        _print_tool_preview("predict_match_score_with_odds", payload)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"一步式预测数据获取失败: {str(e)}"

@tool
def search_football_match_result(query: str) -> str:
    """
    从本地 Qdrant（collection_name='football_match_2026_url'）检索最相关的赛后报告片段。
    注意：使用 qdrant_client 的本地 search 方式，避免 LangChain wrapper 在当前版本不兼容。
    """
    try:
        collection_name = "football_match_2026_url"
        client = QdrantClient(path="local_qdrant")

        qvec = get_embeddings().embed_query(query)
        points = client._client.search(
            collection_name=collection_name,
            query_vector=qvec,
            limit=4,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return "赛后报告库中未检索到相关内容。请先通过 /add_txt 或 /add_urls 写入赛后报告。"

        pieces = []
        for idx, p in enumerate(points, start=1):
            payload = getattr(p, "payload", {}) or {}
            text = payload.get("page_content", "") if isinstance(payload, dict) else ""
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 900:
                text = text[:900] + "...（截断）"

            md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            source = ""
            if isinstance(md, dict):
                source = md.get("source") or md.get("url") or md.get("origin") or ""

            pieces.append(f"【{idx}】来源: {source or '未知'}\n{text}")

        _print_tool_preview("search_football_match_result", {"query": query, "results": len(pieces)})
        return "\n\n".join(pieces)
    except Exception as e:
        return f"赛后报告检索失败: {str(e)}"

class Agent:
    def __init__(self):
        if os.getenv("USE_OLLAMA") == "1":
            from langchain_ollama import ChatOllama
            self.model = ChatOllama(model="qwen3.5:latest")
        else:
            self.model = ChatOpenAI(
                model="deepseek-ai/DeepSeek-V4-Pro",
                api_key=os.getenv("SILICONFLOW_API_KEY", "your_siliconflow_api_key"),
                base_url="https://api.siliconflow.cn/v1",
                temperature=0.7,
            )
        self.memory = SimpleSummaryBuffer(llm=self.model, max_token_limit=2000)
        self.tools = [
            get_weather,
            # multiply,
            get_upcoming_events,
            get_event_odds,
            find_arbitrage_opportunities,
            analyze_match_for_betting,
            predict_match_score_with_odds,
            search_football_match_result,
            search_match_schedule,
            get_team_details,
            search_team_injury,
            store_team_reports,
            search_match_knowledge,
            search_web
        ]
        self.system_prompt = '''你是一个专业的足球比赛分析助手，以下是你的行为规则：
        1. 你会根据用户的问题使用不同的适合的工具来回答问题。
        2. 你会保存每次聊天记录中的重点，以便在后续对话中使用。

        以下是工具与数据规则（不可忽略）：
        A. 当用户问到足球比赛、赔率、下注、比分预测，或请求分析比赛/球队对阵时，必须优先调用工具获取赛事数据后再回答。禁止在未调用工具时凭空分析。
        B. 赛前分析标准流程：
            1) 先调用 search_match_schedule（或 get_upcoming_events）获取赛事基本信息（对阵、时间、场地）；
            2) 再调用 store_team_reports 分别收集双方球队近N场比赛数据并存入知识库；
            3) 调用 search_team_injury 分别查询双方球队的伤病、停赛、预计首发情况；
            4) 然后调用 search_match_knowledge 从知识库检索双方近期表现、战术等详细信息；
            5) 同时调用 get_team_details 获取主教练、关键球员、统计数据；
            6) 最后综合所有数据输出分析。如需赔率数据再调用 get_event_odds。
            store_team_reports 入库耗时较长（需LLM提取实体），每次分析前只需调用一次。
            默认优先使用 Bet365 查询赔率。
        C. 在足球比赛赔率/下注/比分预测场景中，禁止调用 search_web 搜索赔率或编造数据。
            但允许使用 search_match_schedule 获取 get_upcoming_events 不覆盖的赛事赛程信息（如世界杯等杯赛）。
        D. 禁止编造赔率、球队状态、伤停信息。拿不到就明确说拿不到。
        E. 只有当工具报错、超时、或未获取到赛事数据时，才禁止输出具体比分；如果“有赛事无赔率”，允许预测但要标注低置信度。
        F. 默认使用简体中文输出；仅在用户明确要求时切换语言。
        G. 预测比分/下注分析时，使用客观、简洁、可核验的专业表达。
        H. 当用户想了解已经结束的球赛时，优先使用search_football_match_result来获取比赛信息。
        I. 当 get_upcoming_events 返回"未找到赛事"（多因世界杯/欧洲杯等国家队杯赛不在 Odds API 覆盖范围），
            应立即改用 search_match_schedule 获取赛程对阵信息，并根据搜索结果继续分析。'''
        self.agent = create_agent(self.model, tools=self.tools, system_prompt=self.system_prompt)

    def ask_agent(self, question: str) -> str:
        """
        向 agent 提问，自动管理记忆，打印答案与生成速度，并返回最终回答文本。
        """
        try:
            return self._ask_agent_impl(question)
        except Exception:
            import traceback
            traceback.print_exc()
            return "处理请求时发生内部错误，请重试。"

    def _ask_agent_impl(self, question: str) -> str:
        # 获取历史消息
        history = self.memory.load_memory_variables({})['history']
        
        def _is_betting_query(text: str) -> bool:
            """
            仅在“预测/下注/赔率相关”时才视为 betting 场景，纯粹询问赛果/比分结果不受限制。
            """
            if not text:
                return False
            strong_keywords = ["预测", "下注", "赔率", "盘口", "让球", "大小球", "胜平负", "投注", "比分", "比分会", "什么比分", "猜一下"]
            lower = text.lower()
            return any(k in text for k in strong_keywords) or ("odds" in lower) or ("bet" in lower)

        def _has_tool_message(messages) -> bool:
            for m in messages or []:
                cls = m.__class__.__name__.lower()
                if "tool" in cls:
                    return True
            return False

        def _has_tool_name(messages, tool_name: str) -> bool:
            target = (tool_name or "").lower()
            for m in messages or []:
                name = str(getattr(m, "name", "")).lower()
                if target and target in name:
                    return True
                # 兼容部分 ToolMessage 未暴露 name 的情况
                raw = str(getattr(m, "content", "")).lower()
                if target and target in raw:
                    return True
            return False

        def _looks_like_score_answer(text: str) -> bool:
            if not text:
                return False
            has_score = re.search(r"\b\d+\s*[-:]\s*\d+\b", text) is not None
            has_predict_word = ("预测" in text) or ("比分" in text)
            has_risk = "风险" in text
            return has_score and has_predict_word and has_risk

        def _is_result_query(text: str) -> bool:
            """
            判断用户是否在问"赛果/比赛结果/赛后信息"，此类问题应优先走 search_football_match_result。
            """
            if not text:
                return False
            keywords = ["结果", "赛后", "淘汰赛", "半决赛", "结束", "刚结束", "赛果"]
            return any(k in text for k in keywords)

        def _is_match_analysis_query(text: str) -> bool:
            """
            判断是否为比赛分析类问题（分析对阵/球队/攻防等，不含下注/预测关键词）。
            此类问题必须先从工具获取赛事基本信息再分析。
            """
            if not text:
                return False
            if _is_betting_query(text) or _is_result_query(text):
                return False
            analysis_kw = ["分析", "怎么看", "觉得", "评价", "点评", "解读", "介绍", "讲讲", "攻防", "战术", "阵型", "阵容", "伤停"]
            match_kw = ["比赛", "对阵", "这场", "那场", "足球", "球队", "联赛", "决赛", "杯赛", "欧冠", "英超", "西甲", "主场", "客场"]
            has_analysis = any(k in text for k in analysis_kw)
            has_match = any(k in text for k in match_kw)
            has_vs = bool(re.search(r"\bvs\b", text, re.I))
            return (has_analysis and has_match) or (has_vs and not _is_betting_query(text))


        # ======== 分层收集 vs 单轮调用 ========
        start = time.time()
        print(f"[Agent] 开始处理: {question[:80]}...")

        is_analysis = _is_betting_query(question) or _is_match_analysis_query(question)

        if is_analysis:
            # ── 三阶段分层收集 ──
            base_messages = list(history)

            # Stage 1: 赛事基本信息
            print("[Agent] Stage 1/3: 获取赛事基本信息...")
            stage1 = base_messages + [HumanMessage(content=(
                "【阶段1/3：获取赛事基本信息】\n"
                "使用 search_match_schedule 或 get_upcoming_events 获取对阵、时间、场地、阶段。\n"
                "只获取信息，不要做其他操作，不要输出分析。\n\n"
                f"用户问题：{question}"
            ))]
            resp = self.agent.invoke({"messages": stage1})
            t1 = time.time() - start
            print(f"[Agent] Stage 1 完成，耗时 {t1:.1f}s")

            # Stage 2: 球队数据收集
            print("[Agent] Stage 2/3: 收集球队数据...")
            stage2 = list(resp["messages"]) + [HumanMessage(content=(
                "【阶段2/3：收集球队数据】\n"
                "依次执行以下操作，只收集数据不输出分析：\n"
                "1) store_team_reports — 收集双方近期战绩并存入知识库（先主队后客队）\n"
                "2) search_team_injury — 查询双方伤病、停赛、预计首发\n"
                "3) get_team_details — 查询双方主教练、关键球员、统计数据\n"
                "完成所有数据收集后，简单确认已完成。"
            ))]
            resp = self.agent.invoke({"messages": stage2})
            t2 = time.time() - start
            print(f"[Agent] Stage 2 完成，耗时 {t2:.1f}s")

            # Stage 3: 知识检索 + 综合输出
            print("[Agent] Stage 3/3: 知识库检索与综合分析...")
            stage3 = list(resp["messages"]) + [HumanMessage(content=(
                "【阶段3/3：综合分析输出】\n"
                "1) 调用 search_match_knowledge 检索知识库中双方近期表现、历史交锋\n"
                "2) 综合前两阶段收集的所有数据，输出专业比赛分析，必须包含：\n"
                "   - 已获取数据摘要\n"
                "   - 双方实力对比（阵容完整性、近期状态、战术风格）\n"
                "   - 比分预测：给出2-3个具体比分（如 2-0、2-1、1-1）并简要说明理由\n"
                "   - 置信度标注（有赔率→常规置信度，无赔率→低置信度）\n"
                "   - 风险提示\n"
                "禁止再调用 store_team_reports。如需赔率可调 get_event_odds。"
            ))]
            resp = self.agent.invoke({"messages": stage3})
            print(f"[Agent] Stage 3 完成，总耗时 {time.time()-start:.1f}s")

        else:
            # ── 非分析类问题：单轮调用 ──
            resp = self.agent.invoke(
                {"messages": history + [HumanMessage(content=question)]}
            )
            print(f"[Agent] 单轮完成，耗时 {time.time()-start:.1f}s，消息数 {len(resp.get('messages', []))}")

            # 赛果类问题：若未调用 search_football_match_result，则重试
            if _is_result_query(question) and (not _has_tool_name(resp.get("messages", []), "search_football_match_result")):
                print("[Agent] 检测到 result 查询，触发重试...")
                resp = self.agent.invoke(
                    {"messages": history + [HumanMessage(content=question), HumanMessage(content=(
                        "这是已结束比赛的赛果问题。调用 search_football_match_result 获取赛后报告。"
                    ))]}
                )

        # ======== 安全闸门（最终防线）========
        if is_analysis and (not _has_tool_message(resp.get("messages", []))):
            content = "本次未成功调用赛事数据工具，已阻止无数据支撑的分析。请重试。"
            self.memory.save_context({'input': question}, content)
            print(content)
            return content

        elapsed = time.time() - start
        
        def _message_text(msg) -> str:
            raw = getattr(msg, "content", "")
            if isinstance(raw, str):
                return raw.strip()
            if isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, str):
                        txt = item.strip()
                        if txt:
                            parts.append(txt)
                    elif isinstance(item, dict):
                        # 兼容 langchain 常见的 content block 结构
                        txt = item.get("text") or item.get("content") or ""
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                return "\n".join(parts).strip()
            return str(raw).strip() if raw is not None else ""

        # 提取最终回答：优先最后一条非空文本消息，避免返回空字符串
        messages = resp.get("messages", [])
        last_msg = messages[-1] if messages else None
        content = ""
        for msg in reversed(messages):
            content = _message_text(msg)
            if content:
                last_msg = msg
                break

        if not content:
            content = "这次模型没有返回可读文本，请重试一次。"

        content = _strip_agent_output_leak(content)

        # 防止“数据缺失却给具体比分”的幻觉输出
        # 仅在“明确接口失败/超时”时才触发比分保护，避免“已拿到赛事但缺赔率”被误杀
        hard_failure_markers = [
            "调用失败",
            "连接超时",
            "请求超时",
            "read timed out",
            "网络异常",
            "获取赛事失败",
            "比赛分析数据获取失败",
        ]
        score_pattern = r"\b\d+\s*[-:]\s*\d+\b"
        has_hard_failure_signal = any(marker.lower() in content.lower() for marker in hard_failure_markers)
        has_score_prediction = re.search(score_pattern, content) is not None
        if has_hard_failure_signal and has_score_prediction:
            content = (
                "## 已获取数据\n\n"
                "- 工具调用存在失败/超时，当前缺少稳定的实时赛事或赔率数据。\n\n"
                "## 赔率/市场信息\n\n"
                "- 当前未获取到可用赔率数据，无法形成可靠投注依据。\n\n"
                "## 比分预测\n\n"
                "- 暂不具备预测条件（数据不足），本次不输出具体比分。\n\n"
                "## 风险提示\n\n"
                "- 请在拿到实时赔率和球队最新信息后再做判断，可稍后重试。"
            )

        # “有赛事无赔率”时允许预测，但自动降级置信度提示
        no_odds_markers = [
            "未获取到赔率",
            "未获取到 bet365 赔率",
            "当前未获取到可用赔率数据",
            "无法提供实时赔率",
        ]
        has_no_odds_signal = any(marker.lower() in content.lower() for marker in no_odds_markers)
        has_confidence_tag = ("置信度" in content) or ("低置信度" in content)
        if (not has_hard_failure_signal) and has_no_odds_signal and has_score_prediction and (not has_confidence_tag):
            content += (
                "\n\n## 置信度\n\n"
                "- 低置信度：当前缺少实时赔率，本次比分预测主要基于赛事基础信息与一般对阵强弱，参考性有限。"
            )
        
        # 语言守卫：若输出英文占比过高，自动改写为简体中文
        def _needs_zh_rewrite(text: str) -> bool:
            if not text:
                return False
            zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
            en_chars = re.findall(r"[A-Za-z]", text)
            # 英文字母显著多于中文时，判定发生语言漂移
            return len(en_chars) > max(120, len(zh_chars) * 2)

        if _needs_zh_rewrite(content):
            try:
                rewrite_prompt = (
                    "请将以下内容改写为简体中文，保持原始事实与结构，不要新增虚构信息，"
                    "保留项目符号和层次：\n\n" + content
                )
                rewritten = self.model.invoke(rewrite_prompt)
                rewritten_text = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
                if isinstance(rewritten_text, str) and rewritten_text.strip():
                    content = rewritten_text.strip()
            except Exception:
                # 改写失败时保留原内容，避免影响主流程
                pass

        # 最终输出强制器：预测类问题若仍未给出具体比分，则改写为可直接使用的预测格式
        if _is_betting_query(question) and (not _looks_like_score_answer(content)):
            try:
                enforce_prompt = (
                    "你是足球分析助手。请将下面内容改写为最终答复，严格满足：\n"
                    "1) 必须包含四段：已获取数据、赔率/市场信息、比分预测、风险提示；\n"
                    "2) 比分预测必须给出2-3个具体比分（如1-1、1-2、2-1）；\n"
                    "3) 如果赔率缺失，明确标注“低置信度”，但仍需给出比分；\n"
                    "4) 禁止输出“你想让我做什么/下一步”等反问；\n"
                    "5) 使用简体中文，语气专业简洁。\n\n"
                    f"用户问题：{question}\n\n"
                    f"待改写内容：\n{content}"
                )
                enforced = self.model.invoke(enforce_prompt)
                enforced_text = enforced.content if hasattr(enforced, "content") else str(enforced)
                if isinstance(enforced_text, str) and enforced_text.strip():
                    content = enforced_text.strip()
            except Exception:
                pass

        # 保存记忆
        self.memory.save_context({'input': question}, content)
        
        # 打印最终答案
        # print(content)
        
        # ---------- 速度测量 ----------
        token_info = getattr(last_msg, "response_metadata", {}) if last_msg else {}
        output_tokens = token_info.get("eval_count", 0)
        
        # # 在速度信息前空一行
        # print()  
        if output_tokens:
            gen_speed = output_tokens / elapsed
            print(f"生成 token 数：{output_tokens}, 耗时：{elapsed:.2f}秒, 生成速度：{gen_speed:.2f} tokens/s")
        
        # 查看摘要记忆
        if self.memory.buffer:
            try:
                print('\n以下是记忆内容:')
                print(self.memory.buffer)
            except UnicodeEncodeError:
                pass

        return content

    # def emotion_chain(self, question: str) -> str:

    def background_voice_synthesis(self, text:str, uid:str):
        # 不需要返回值，触发语音合成
        asyncio.run(self.get_voice(text, uid))

    async def get_voice(self, text: str, uid: str):
        """使用 Microsoft Edge 在线 TTS（edge-tts 包），等价于：
        edge-tts --voice zh-CN-XiaoxiaoNeural --text <text> --write-media voice/<uid>.mp3
        """
        raw = (text or "").strip()
        msg = _speech_text_for_tts(raw)
        print("test2speech", msg)
        print("uid: ", uid)
        if not msg:
            print("edge-tts: 跳过空文本（清洗后无对白）")
            return
            
        base = os.path.dirname(os.path.abspath(__file__))
        voice_dir = os.path.join(base, "voice")
        os.makedirs(voice_dir, exist_ok=True)
        out_path = os.path.join(voice_dir, f"{uid}.mp3")
        voice = "zh-CN-XiaoxiaoNeural"
        try:
            communicate = edge_tts.Communicate(msg, voice)
            await communicate.save(out_path)
            print(f"edge-tts 已写入: {out_path}")
        except Exception as e:
            print(f"edge-tts 合成失败: {e}")
        
agent = Agent()

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name='bge-small-zh-v1.5',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
        )
    return _embeddings


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/qq_voice/{file_id}")
def qq_voice_public_file(file_id: str):
    """
    供 QQ 服务端拉取 mp3（post_c2c_file / post_group_file 的 url 须为公网 HTTPS）。
    仅允许标准 uuid 文件名，映射到本目录 voice/。
    """
    if not re.fullmatch(r"[0-9a-fA-F-]{36}\.mp3", file_id, flags=re.I):
        raise HTTPException(status_code=404, detail="not found")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "voice", file_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="audio/mpeg", filename=file_id)


@app.post("/chat")
def chat(query: str, background_tasks: BackgroundTasks):
    msg = agent.ask_agent(query)
    uni_id = str(uuid.uuid4())
    background_tasks.add_task(agent.background_voice_synthesis, msg, uni_id)
    return {"msg": msg, "id": uni_id}


def _check_local_agent_api_key(request: Request) -> None:
    expected = os.getenv("LOCAL_AGENT_API_KEY")
    if not expected:
        return
    auth = request.headers.get("authorization") or ""
    if auth.strip() != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/v1/models")
def openai_list_models(request: Request):
    """给 OpenClaw / 其它 OpenAI 兼容客户端做模型发现。"""
    _check_local_agent_api_key(request)
    mid = os.getenv("LOCAL_AGENT_MODEL_ID", "local-agent")
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(request: Request, body: dict):
    """
    OpenAI Chat Completions 兼容接口：请求体会被转成一段文本后调用 ask_agent。
    供 OpenClaw 等把「自定义 / 本地 API」指到 http://<本机>:8000/v1
    （不设 LOCAL_AGENT_API_KEY 时不校验 Authorization）。
    """
    _check_local_agent_api_key(request)
    if body.get("stream"):
        raise HTTPException(
            status_code=400,
            detail="暂不支持 stream=true，请在网关侧关闭流式或使用非流式",
        )
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages 须为非空数组")
    question = _openai_messages_to_question(messages)
    if not question.strip():
        raise HTTPException(status_code=400, detail="无有效 user/system/assistant 文本")
    reply = agent.ask_agent(question.strip())
    mid = body.get("model") or os.getenv("LOCAL_AGENT_MODEL_ID", "local-agent")
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    ts = int(time.time())
    return {
        "id": cid,
        "object": "chat.completion",
        "created": ts,
        "model": mid,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/add_txt")
def add_txt(file_path: str):
    """
    直接把本地 .txt（赛后报告）写入 Qdrant，用于后续检索：
    - 通常你先问“赛后有什么关键点”，agent 会调用 `search_football_match_result`
    """
    collection_name = "football_match_2026_url"
    if not file_path:
        return {"response": "file_path 不能为空"}

    # 安全兜底：仅允许读取当前目录或其子目录
    base_dir = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(file_path)
    if not abs_path.lower().startswith(base_dir.lower()):
        return {"response": "不允许读取工作区之外的文件"}

    if not os.path.exists(abs_path):
        return {"response": f"文件不存在：{abs_path}"}

    try:
        try:
            loader = TextLoader(abs_path, encoding="utf-8")
        except UnicodeDecodeError:
            # 中文文件常见编码回退
            loader = TextLoader(abs_path, encoding="gb18030")

        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
        ).split_documents(docs)

        client = QdrantClient(path="local_qdrant")
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            from qdrant_client.models import VectorParams, Distance
            emb = get_embeddings()
            dim = len(emb.embed_query("vector_dim_probe"))
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        qdrant = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=get_embeddings(),
        )
        qdrant.add_documents(documents)
        print("向量数据库创建/追加完成：add_txt")
        return {"response": "TXT added", "chunks": len(documents)}
    except Exception as e:
        return {"response": f"add_txt 失败: {str(e)}"}


@app.post("/add_urls")
def add_urls(URL: str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)
    # 引入向量数据库
    client = QdrantClient(path="local_qdrant")
    # 使用 add_documents 追加前，确保 collection 存在
    collection_name = "football_match_2026_url"
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        # 如果是本地 qdrant 且集合不存在，这里需要手动创建
        from qdrant_client.models import VectorParams, Distance

        emb = get_embeddings()
        # 取向量维度（bge-small 输出维度通常为 384，但用真实计算确保兼容）
        dim = len(emb.embed_query("vector_dim_probe"))
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=get_embeddings(),
    )
    qdrant.add_documents(documents)
    print("向量数据库创建完成")
    return {"response": "URLs added"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added"}

@app.post("/add_tests")
def add_tests():
    return {"response": "Tests added"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)