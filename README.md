# Soccer AI Agent ⚽

基于 LLM Agent 的智能足球赛前分析系统。通过多源实时数据采集、本地知识图谱构建、三阶段分层推理，实现从信息收集到比分预测的全流程自动化。

## 架构概览

```
用户 (QQ消息)
  │
  ▼
qq_agent_bridge.py ── WebSocket ──→ QQ 服务器
  │ asyncio.to_thread()
  ▼
Agent.ask_agent() 
  │
  ├─ Stage 1: search_match_schedule   → 获取对阵/时间/场地
  ├─ Stage 2: store_team_reports      → 近期战绩入库 (LightRAG)
  │           search_team_injury      → 伤病/停赛/首发
  │           get_team_details        → 教练/球员/统计
  └─ Stage 3: search_match_knowledge  → 知识库检索
              LLM 综合输出           → 比分预测+分析
  │
  ├── 数据源: SerpAPI / Odds API / 天气 API
  ├── 大模型: DeepSeek-V4-Pro (对话) + V4-Flash (实体提取)
  └── 知识库: LightRAG 图数据库 (本地 bge-small-zh-v1.5 嵌入)
```

## 特性

- **三阶段分层推理**：赛程→球队数据→综合分析，逐层递进，避免 context 过载
- **知识图谱**：LightRAG 自动从赛后报告提取实体关系，支持图+向量混合检索
- **8 个专用工具**：Agent 自主编排调用，单一职责设计
- **安全闸门**：多级校验防止 LLM 在无数据时编造信息
- **双模式切换**：SiliconFlow API (默认) / 本地 Ollama，通过环境变量切换
- **QQ 机器人**：文字回复 + 可选的语音回复 (Edge-TTS)
- **OpenAI 兼容接口**：可对接其他客户端 (`/v1/chat/completions`)

## 快速开始

### 1. 环境准备

```bash
conda create -n agent_env python=3.12
conda activate agent_env
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

| 必需 | 说明 | 获取地址 |
|------|------|---------|
| `SILICONFLOW_API_KEY` | 大模型 API | https://siliconflow.cn |
| `SERPAPI_API_KEY` | 网页搜索 | https://serpapi.com |
| `QQ_BOT_APP_ID` | QQ 机器人 AppID | https://q.qq.com |
| `QQ_BOT_APP_SECRET` | QQ 机器人密钥 | https://q.qq.com |

### 3. 下载嵌入模型（本地）

```bash
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5
# 或: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

### 4. 启动

```bash
# 完整启动 (server + 隧道 + QQ bot)
python run_qq_voice_b.py

# 仅启动 API 服务 (调试用)
python server.py

# 关闭语音
$env:QQ_BOT_SEND_VOICE="0"; python run_qq_voice_b.py
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `python run_qq_voice_b.py` | 一键启动（推荐） |
| `python server.py` | 仅启动 FastAPI |
| `python qq_agent_bridge.py` | 仅启动 QQ 桥接 |
| `$env:USE_OLLAMA="1"; python run_qq_voice_b.py` | 切换本地 Ollama 模式 |
| `$env:QQ_BOT_SEND_VOICE="0"; python run_qq_voice_b.py` | 关闭语音 |

## 工作流程

### Stage 1: 赛事基本信息

Agent 调用 `search_match_schedule` 获取对阵双方、时间、场地、比赛阶段。

### Stage 2: 球队数据收集

- `store_team_reports` → 搜索双方近期战绩 → 存入 LightRAG 知识库
- `search_team_injury` → 搜索伤病/停赛/预计首发
- `get_team_details` → 获取主教练、关键球员、统计数据

### Stage 3: 综合分析输出

- `search_match_knowledge` → 从 LightRAG 知识库检索近期表现
- LLM 综合所有数据，输出：实力对比 → 比分预测(2-3个) → 置信度标注 → 风险提示

## 项目结构

```
.
├── server.py              # FastAPI 服务 + Agent 核心
├── lightrag_kb.py         # LightRAG 知识库封装
├── qq_agent_bridge.py     # QQ 机器人桥接
├── run_qq_voice_b.py      # 一键启动脚本
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
├── .gitignore
├── city_cn.csv            # 国内城市编码
├── city_in.csv            # 国际城市编码
└── README.md
```

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek-V4-Pro / V4-Flash (SiliconFlow) |
| Agent 框架 | LangChain + LangGraph |
| 知识图谱 | LightRAG + NanoVectorDB |
| 嵌入模型 | bge-small-zh-v1.5 (本地 512维) |
| 后端 | FastAPI + Uvicorn |
| 数据源 | SerpAPI, Odds API, 天气 API |
| IM | QQ 机器人 (botpy) |
| 语音 | Edge-TTS |
| 部署 | Windows / cloudflared 隧道 |

## License

MIT
