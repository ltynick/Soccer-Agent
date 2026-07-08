# -*- coding: utf-8 -*-
"""
执行「方案 B」：本机起 FastAPI → 公网隧道暴露 8000 → 自动设置 QQ_VOICE_PUBLIC_BASE → 启动 qq_agent_bridge。

隧道优先级:
  1) cloudflared（trycloudflare.com）— QQ 服务端能稳定拉取 mp3，推荐
  2) localtunnel（loca.lt）— 易触发 QQ 40002 download file err，仅作备选

前置:
  在项目根目录编辑 qq_env.py，填入 QQ_BOT_APP_ID / QQ_BOT_APP_SECRET（已自动 import）；
  或仍可用系统环境变量覆盖。

依赖:
  cloudflared（推荐）: winget install Cloudflare.cloudflared
  或 Node.js npx（备选 localtunnel）
  pip: edge-tts（server 合成语音用）

用法（在项目根目录）:
  python run_qq_voice_b.py

结束: 桥接进程退出后会自动关掉隧道与 server。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL = "http://127.0.0.1:8000/"


def _wait_server(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(LOCAL, timeout=2)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError("等待 server.py 监听 8000 超时")


def _find_cloudflared() -> str | None:
    exe = shutil.which("cloudflared")
    if exe:
        return exe
    if sys.platform == "win32":
        for cand in (
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
            r"C:\Program Files\Cloudflare\cloudflared\cloudflared.exe",
        ):
            if os.path.isfile(cand):
                return cand
    return None


def _popen_localtunnel() -> subprocess.Popen:
    """Windows 上 npx 是 .cmd，列表形式 Popen 常报 WinError 2，需走 shell。"""
    common = dict(
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if sys.platform == "win32":
        return subprocess.Popen(
            "npx -y localtunnel --port 8000",
            shell=True,
            **common,
        )
    return subprocess.Popen(
        ["npx", "-y", "localtunnel", "--port", "8000"],
        **common,
    )


def _parse_tunnel_url(line: str, kind: str) -> str | None:
    if kind == "cloudflared":
        m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line, re.I)
        if m:
            return m.group(1).strip().rstrip("/")
        return None
    m = re.search(r"your url is:\s*(https://\S+)", line, re.I)
    if m:
        return m.group(1).strip().rstrip("/")
    # 部分版本 cloudflared 日志混在输出里，顺带再扫 trycloudflare
    m2 = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line, re.I)
    return m2.group(1).strip().rstrip("/") if m2 else None


def main() -> int:
    try:
        import qq_env  # noqa: F401 — 个人凭证，见 qq_env.py
    except ImportError:
        pass
    if not os.getenv("QQ_BOT_APP_ID") or not os.getenv("QQ_BOT_APP_SECRET"):
        print("请先设置环境变量 QQ_BOT_APP_ID 与 QQ_BOT_APP_SECRET（或编辑 qq_env.py）。")
        return 1

    os.chdir(ROOT)
    server = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    public_base: str | None = None
    tunnel_proc: subprocess.Popen | None = None
    tunnel_kind: str = ""

    try:
        _wait_server()
        if server.poll() is not None:
            print("server.py 已退出（可能 8000 被占用或启动失败），请检查终端/端口。")
            return 1

        cf = _find_cloudflared()
        if cf:
            tunnel_kind = "cloudflared"
            print("使用 cloudflared（推荐，QQ 可拉取 mp3）…")
            tunnel_proc = subprocess.Popen(
                [cf, "tunnel", "--url", "http://127.0.0.1:8000"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            if not shutil.which("npx"):
                print("未找到 cloudflared 与 npx。")
                print("  推荐: winget install Cloudflare.cloudflared 后重试。")
                print("  或安装 Node.js 以使用 npx localtunnel（QQ 语音易失败）。")
                return 1
            tunnel_kind = "localtunnel"
            print(
                "警告: 使用 localtunnel（loca.lt）。QQ 服务端常无法下载 mp3，"
                "报 40002 download file err；请安装 cloudflared 后重试本脚本。\n"
                "正在建立 localtunnel …"
            )
            tunnel_proc = _popen_localtunnel()

        assert tunnel_proc is not None and tunnel_proc.stdout is not None
        for _ in range(240):
            line = tunnel_proc.stdout.readline()
            if not line:
                break
            print(line, end="")
            public_base = _parse_tunnel_url(line, tunnel_kind)
            if public_base:
                break
        if not public_base:
            print("未能解析隧道公网地址，请手动运行 cloudflared 并设置 QQ_VOICE_PUBLIC_BASE。")
            return 1

        env = os.environ.copy()
        env["QQ_VOICE_PUBLIC_BASE"] = public_base
        print(f"\n已设置 QQ_VOICE_PUBLIC_BASE={public_base}")
        print("启动 qq_agent_bridge（结束后会自动关掉 server 与隧道）…\n")

        bridge = subprocess.run(
            [sys.executable, "qq_agent_bridge.py"],
            cwd=ROOT,
            env=env,
        )
        return bridge.returncode
    finally:
        if tunnel_proc is not None:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
            try:
                tunnel_proc.wait(timeout=5)
            except Exception:
                try:
                    tunnel_proc.kill()
                except Exception:
                    pass
        try:
            server.terminate()
        except Exception:
            pass
        try:
            server.wait(timeout=5)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main() or 0)
