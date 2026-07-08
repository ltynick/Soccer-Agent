# -*- coding: utf-8 -*-
"""
LightRAG 知识库模块 — 赛后报告存储与检索
用小模型 qwen2.5:1.5b 做实体提取，本地 bge-small-zh-v1.5 做嵌入
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import asyncio
import httpx
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.join(ROOT, "match_knowledge")

_rag_instance = None
_embeddings = None
_init_loop = None
_init_lock = asyncio.Lock()


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=os.path.join(ROOT, "bge-small-zh-v1.5"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
        )
    return _embeddings


async def _embed_async(texts):
    result = await asyncio.to_thread(_get_embeddings().embed_documents, texts)
    return np.array(result)


async def _ollama_extract(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:1.5b",
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 4096, "temperature": 0.0},
            },
        )
        return resp.json()["message"]["content"]


async def _siliconflow_chat(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs,
) -> str:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "your_siliconflow_api_key")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "messages": messages,
                "stream": False,
                "temperature": 0.0,
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _build_rag():
    if os.environ.get("USE_OLLAMA") == "1":
        llm_func = _ollama_extract
        llm_name = "qwen2.5:1.5b"
    else:
        llm_func = _siliconflow_chat
        llm_name = "deepseek-ai/DeepSeek-V4-Flash"

    return LightRAG(
        working_dir=WORKDIR,
        llm_model_func=llm_func,
        llm_model_name=llm_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=512,
            max_token_size=512,
            func=_embed_async,
        ),
        chunk_token_size=800,
        chunk_overlap_token_size=50,
        entity_extract_max_gleaning=0,
        llm_model_max_async=2,
    )


async def _ensure_init():
    global _rag_instance, _init_loop
    async with _init_lock:
        if _rag_instance is not None:
            return
        os.makedirs(WORKDIR, exist_ok=True)
        print("[LightRAG] 正在初始化知识库...")
        _rag_instance = _build_rag()
        await _rag_instance.initialize_storages()
        _init_loop = asyncio.get_running_loop()
        print("[LightRAG] 知识库初始化完成")


def get_rag():
    return _rag_instance


def _run_on_init_loop(coro, timeout: float = 600.0):
    """将异步操作调度到初始化事件循环上执行（用于从线程中调用）"""
    if _init_loop is None:
        raise RuntimeError("LightRAG 尚未初始化")
    future = asyncio.run_coroutine_threadsafe(coro, _init_loop)
    return future.result(timeout=timeout)


def insert_knowledge_sync(text: str) -> bool:
    rag = _rag_instance
    if rag is None:
        print("[LightRAG] 尚未初始化，跳过插入")
        return False
    try:
        _run_on_init_loop(rag.ainsert(text))
        return True
    except Exception as e:
        print(f"[LightRAG] 插入失败: {e}")
        return False


def search_knowledge_sync(query: str, top_k: int = 8) -> str:
    rag = _rag_instance
    if rag is None:
        return "知识库尚未初始化。"
    try:
        return _run_on_init_loop(rag.aquery(query, param=QueryParam(mode="hybrid", top_k=top_k)))
    except Exception as e:
        return f"知识库检索失败: {e}"
