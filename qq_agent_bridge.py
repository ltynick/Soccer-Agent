# -*- coding: utf-8 -*-
"""
用本地 botpy 源码（botpy-master）连 QQ，回复由 server.Agent.ask_agent 生成。
可选：在文字回复后再发一条 QQ 语音气泡；mp3 仍由 server 里与 /chat 相同的
`background_voice_synthesis` → `get_voice` 写入 voice/，需公网 HTTPS 暴露 /qq_voice/（见下文）。

用法（在项目根目录 agent 下）:
  编辑 qq_env.py 填入 AppID/AppSecret；或 set QQ_BOT_APP_ID / QQ_BOT_APP_SECRET
  python qq_agent_bridge.py

语音（可选）:
  1) 另开终端运行: python server.py（或 uvicorn），保证可通过公网 HTTPS 访问
     例如 https://你的域名/qq_voice/<uuid>.mp3
  2) set QQ_VOICE_PUBLIC_BASE=https://你的域名
     （不要末尾斜杠；与 server 对外域名一致，供 QQ 服务器拉取 mp3）
  3) 关闭语音: set QQ_BOT_SEND_VOICE=0
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BOTPY_SRC = os.path.join(_ROOT, "botpy-master")
if _BOTPY_SRC not in sys.path:
    sys.path.insert(0, _BOTPY_SRC)

import botpy  # noqa: E402
from botpy import logging  # noqa: E402
from botpy.message import C2CMessage, GroupMessage  # noqa: E402

_log = logging.get_logger()

# QQ 单条文本过大会失败，保守截断
_QQ_REPLY_MAX_LEN = 3800
# 语音只念前若干字，避免合成过久
_QQ_TTS_MAX_CHARS = 25000


def _truncate(s: str) -> str:
    s = s or ""
    if len(s) <= _QQ_REPLY_MAX_LEN:
        return s
    return s[: _QQ_REPLY_MAX_LEN - 1] + "…"


def _voice_enabled() -> bool:
    if os.getenv("QQ_BOT_SEND_VOICE", "1").strip() == "0":
        return False
    return bool((os.getenv("QQ_VOICE_PUBLIC_BASE") or "").strip())


async def _ask_agent_text(question: str) -> str:
    from server import agent

    return await asyncio.to_thread(agent.ask_agent, question)


async def _post_voice_c2c(message: C2CMessage, reply: str) -> None:
    if not _voice_enabled():
        return
    base = os.getenv("QQ_VOICE_PUBLIC_BASE", "").strip().rstrip("/")
    from server import agent as srv_agent

    chunk = (reply or "")[:_QQ_TTS_MAX_CHARS]
    if not chunk.strip():
        return
    uid = str(uuid.uuid4())
    try:
        # 与 /chat 一致：走 background_voice_synthesis → voice/<uid>.mp3
        await asyncio.to_thread(srv_agent.background_voice_synthesis, chunk, uid)
    except Exception as e:
        _log.warning("语音合成跳过: %s", e)
        return

    # trycloudflare 落盘与边缘可达略慢；QQ 拉 mp3 也可能 >5s（botpy 默认 HTTP 超时 5s 会误伤）
    await asyncio.sleep(2.0)

    mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice", f"{uid}.mp3")
    if not os.path.isfile(mp3_path):
        _log.warning(
            "未找到 %s（多为清洗后无对白、edge-tts 未写入），跳过 QQ 语音上传，避免 40002",
            mp3_path,
        )
        return

    api = message._api
    oid = message.author.user_openid
    mid = message.id
    eid = message.event_id
    url = f"{base}/qq_voice/{uid}.mp3"
    try:
        media = await api.post_c2c_file(openid=oid, file_type=3, url=url, srv_send_msg=False)
        if not media or not isinstance(media, dict) or not media.get("file_info"):
            _log.warning("post_c2c_file 未返回有效 media（多为 HTTP 超时），已跳过语音条；请确认 Client(timeout=…) 已加大")
            return
        kw = dict(
            openid=oid,
            msg_type=7,
            msg_id=mid,
            msg_seq=2,
            media=media,
        )
        if eid is not None:
            kw["event_id"] = eid
        await api.post_c2c_message(**kw)
    except Exception as e:
        err = str(e)
        if "download file" in err or "40002" in err:
            _log.warning(
                "QQ 语音发送失败（服务端拉 mp3 失败）: %s — 常见原因：1) 本地未生成 mp3 或 URL 404；"
                "2) loca.lt 隧道；请优先 cloudflared + *.trycloudflare.com",
                e,
            )
        elif "invalid file_info" in err or "304080" in err:
            _log.warning(
                "QQ 语音发送失败（file_info 无效，常为 post_c2c_file 超时后仍发富媒体）: %s",
                e,
            )
        else:
            _log.warning("QQ 语音消息发送失败: %s", e)


async def _post_voice_group(message: GroupMessage, reply: str) -> None:
    if not _voice_enabled():
        return
    base = os.getenv("QQ_VOICE_PUBLIC_BASE", "").strip().rstrip("/")
    from server import agent as srv_agent

    chunk = (reply or "")[:_QQ_TTS_MAX_CHARS]
    if not chunk.strip():
        return
    uid = str(uuid.uuid4())
    try:
        await asyncio.to_thread(srv_agent.background_voice_synthesis, chunk, uid)
    except Exception as e:
        _log.warning("语音合成跳过: %s", e)
        return

    await asyncio.sleep(2.0)

    mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice", f"{uid}.mp3")
    if not os.path.isfile(mp3_path):
        _log.warning("未找到 %s，跳过 QQ 群语音上传", mp3_path)
        return

    api = message._api
    gid = message.group_openid
    mid = message.id
    eid = message.event_id
    url = f"{base}/qq_voice/{uid}.mp3"
    try:
        media = await api.post_group_file(group_openid=gid, file_type=3, url=url, srv_send_msg=False)
        if not media or not isinstance(media, dict) or not media.get("file_info"):
            _log.warning("post_group_file 未返回有效 media（多为 HTTP 超时），已跳过语音条")
            return
        kw = dict(
            group_openid=gid,
            msg_type=7,
            msg_id=mid,
            msg_seq=2,
            media=media,
        )
        if eid is not None:
            kw["event_id"] = eid
        await api.post_group_message(**kw)
    except Exception as e:
        err = str(e)
        if "download file" in err or "40002" in err:
            _log.warning(
                "QQ 群语音拉 mp3 失败: %s — 检查本地是否已生成 mp3、公网 URL 与 cloudflared",
                e,
            )
        elif "invalid file_info" in err or "304080" in err:
            _log.warning("QQ 语音发送失败（file_info 无效）: %s", e)
        else:
            _log.warning("QQ 语音消息发送失败: %s", e)


class AgentQQClient(botpy.Client):
    async def on_ready(self):
        _log.info("robot 「%s」 ready", self.robot.name)
        if _voice_enabled():
            _log.info("语音已启用，QQ_VOICE_PUBLIC_BASE=%s", os.getenv("QQ_VOICE_PUBLIC_BASE", "").strip()[:48])
        else:
            _log.info("语音未启用（未设置 QQ_VOICE_PUBLIC_BASE 或 QQ_BOT_SEND_VOICE=0）")

    async def on_c2c_message_create(self, message: C2CMessage):
        text = (message.content or "").strip()
        if not text:
            return
        try:
            reply = _truncate(await _ask_agent_text(text))
        except Exception as e:
            _log.exception("ask_agent failed: %s", e)
            reply = f"处理出错：{e!s}"[:500]
        api = message._api
        oid = message.author.user_openid
        mid = message.id
        eid = message.event_id
        if reply:
            try:
                print(f"[bridge] 文字回复长度: {len(reply)}, 前80字: {reply[:80]}")
            except UnicodeEncodeError:
                print(f"[bridge] 文字回复长度: {len(reply)}, 前80字: {reply[:80].encode('ascii','replace').decode('ascii')}")
        else:
            print("[bridge] 警告: 文字回复为空！")
        try:
            kw = dict(openid=oid, msg_type=0, msg_id=mid, msg_seq=1, content=reply)
            if eid is not None:
                kw["event_id"] = eid
            await api.post_c2c_message(**kw)
            print("[bridge] 文字消息已发送")
        except Exception as e:
            try:
                print(f"[bridge] 文字消息发送失败: {e}")
            except UnicodeEncodeError:
                print(f"[bridge] 文字消息发送失败: {str(e).encode('ascii','replace').decode('ascii')}")
        await _post_voice_c2c(message, reply)

    async def on_group_at_message_create(self, message: GroupMessage):
        text = (message.content or "").strip()
        if not text:
            return
        try:
            reply = _truncate(await _ask_agent_text(text))
        except Exception as e:
            _log.exception("ask_agent failed: %s", e)
            reply = f"处理出错：{e!s}"[:500]
        api = message._api
        gid = message.group_openid
        mid = message.id
        eid = message.event_id
        kw = dict(group_openid=gid, msg_type=0, msg_id=mid, msg_seq=1, content=reply)
        if eid is not None:
            kw["event_id"] = eid
        await api.post_group_message(**kw)
        await _post_voice_group(message, reply)


def main() -> None:
    try:
        import qq_env  # noqa: F401 — 个人凭证，见 qq_env.py
    except ImportError:
        pass
    appid = os.getenv("QQ_BOT_APP_ID", "").strip()
    secret = os.getenv("QQ_BOT_APP_SECRET", "").strip()
    if not appid or not secret:
        print("请设置环境变量 QQ_BOT_APP_ID 与 QQ_BOT_APP_SECRET 后再运行。")
        sys.exit(1)

    import server as _server  # noqa: F401 — 启动前加载 agent，失败尽早暴露

    if not getattr(_server, "agent", None):
        print("server 模块未暴露 agent，请检查 server.py。")
        sys.exit(1)

    # 初始化 LightRAG 知识库（bridge 模式下无 uvicorn startup event）
    from lightrag_kb import _ensure_init
    print("正在初始化 LightRAG 知识库...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_ensure_init())
    print("LightRAG 知识库就绪。")

    intents = botpy.Intents(public_messages=True)
    # QQ 拉取 trycloudflare 上 mp3 常 >5s；botpy 默认 HTTP 超时 5 会导致 post_*_file 超时 → invalid file_info
    client = AgentQQClient(intents=intents, timeout=120)
    client.run(appid=appid, secret=secret)


if __name__ == "__main__":
    main()
