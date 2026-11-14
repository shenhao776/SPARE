#!/usr/bin/env python


import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.chdir(ROOT)

BUFFER_SIZE = 8192


class _Tee:

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, data):
        self.stream.write(data)
        try:
            self.fh.write(data)
        except Exception:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()


def _load_service(mode):

    from core.vpr_core import VPRService

    service = VPRService()
    active = getattr(service, "reranking_model", None) is not None
    return service, active


def _ensure_live_dataset(scene):

    live = os.path.join(ROOT, "realworld", "live", scene)
    os.makedirs(os.path.join(live, "queries_test"), exist_ok=True)
    db_link = os.path.join(live, "database")
    real_db = os.path.join(ROOT, "datasets", scene, "database")
    if not os.path.isdir(real_db):
        raise SystemExit(
            f"找不到 {real_db} —— 先建这个场景的数据:\n"
            f"  python reproduce.py --scene {scene} --stage build --splits test"
        )
    if os.path.islink(db_link):
        if os.path.realpath(db_link) != os.path.realpath(real_db):
            os.unlink(db_link)
            os.symlink(real_db, db_link)
    elif os.path.exists(db_link):
        raise SystemExit(f"{db_link} 存在但不是软链,请人工确认后删除")
    else:
        os.symlink(real_db, db_link)
    return live


def _spawn_webapp(webapp_root, vpr_port, live_dataset):

    server_py = os.path.join(webapp_root, "web_server.py")
    if not os.path.isfile(server_py):
        server_py = os.path.join(webapp_root, "tools", "web_server.py")
    if not os.path.isfile(server_py):
        raise SystemExit(
            f"找不到 {server_py}\n"
            f"  用 --webapp-root 指向 LookItsDifferent 的 VPR 根目录"
        )
    env = {
        **os.environ,
        "EXTERNAL_VPR_SERVER": "1",
        "VPR_DATASET_PATH": live_dataset,
        "VPR_EXTERNAL_ROOT": ROOT,
    }
    print(f"[star_server] 拉起手机界面: {server_py}", flush=True)
    print(f"[star_server] 它会连本服务的 localhost:{vpr_port}", flush=True)
    return subprocess.Popen(
        [sys.executable, server_py],
        cwd=webapp_root,
        env=env,
    )


def _banner(mode, active, port):
    print("=" * 68, flush=True)
    print(f"  SPARE 真机定位服务   默认 mode={mode}   port={port}", flush=True)
    print(
        "  算法逐请求可切:手机端选 raw / star,请求不带 mode 时用上面的默认值",
        flush=True,
    )
    if active:
        print("  重排:✅ 已加载,star 请求的响应里 rerank_active=true", flush=True)
    else:
        print("  重排:❌ **不可用**,选 star 也等价于 raw!", flush=True)
        print(
            "       通常是权重没找到 —— 核对通用头文件是否在 assets/ 下(见 README)。",
            flush=True,
        )
        print("       响应里会带 rerank_active=false,请据此标注实验记录。", flush=True)
    print("=" * 68, flush=True)


def handle_client(conn, service, default_mode, active):

    try:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            return
        params = json.loads(data.decode("utf-8"))
        mode = str(params.pop("mode", default_mode) or default_mode).lower()
        if mode == "fol":
            conn.sendall(
                json.dumps(
                    {
                        "error": "unsupported mode: fol",
                        "mode": "fol",
                        "rerank_active": False,
                    }
                ).encode("utf-8")
            )
            return
        if mode not in ("raw", "star"):
            mode = default_mode
        params["rerank"] = mode == "star"
        started = time.perf_counter()
        result = service.run_inference(SimpleNamespace(**params))
        if not result:
            conn.sendall(b"OK")
            return
        result["mode"] = mode
        result["rerank_active"] = bool(active and mode == "star")
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        conn.sendall(json.dumps(result).encode("utf-8"))
    except Exception as exc:
        print(f"[star_server] 请求处理失败: {type(exc).__name__}: {exc}", flush=True)
        try:
            conn.sendall(b"OK")
        except OSError:
            pass
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("raw", "star"),
        default="star",
        help="请求不带 mode 时的默认算法。raw=不重排(对照组);" "star=SPARE 重排",
    )
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument(
        "--host",
        default="localhost",
        help="默认只听本机 —— webapp 与本服务同机。真要跨机再放开,"
        "但这条链路全程明文,别暴露到公网",
    )
    ap.add_argument(
        "--with-webapp",
        action="store_true",
        help="同时拉起 LookItsDifferent 的手机界面(web_server.py),一个命令搞定。"
        "退出时一并清理。不加则只起定位服务,webapp 需自己启动",
    )
    ap.add_argument(
        "--webapp-root",
        default=os.path.expanduser("~/shared_files/Point-and-Shoot/VPR"),
        help="web_server.py 所在的 VPR 根目录",
    )
    a = ap.parse_args()

    log_dir = os.path.join(a.webapp_root, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, os.path.join(log_dir, "star_server.log"))
        sys.stderr = sys.stdout
        print(
            f"[star_server] 日志同时写到 {os.path.join(log_dir, 'star_server.log')}",
            flush=True,
        )
    except OSError as exc:
        print(f"[star_server] 日志文件建不了({exc}),只写终端", flush=True)

    service, active = _load_service(a.mode)
    _banner(a.mode, active, a.port)

    from core.config import TARGET_SCENE

    live_dataset = _ensure_live_dataset(TARGET_SCENE)
    print(f"[star_server] live query 目录: {live_dataset}", flush=True)
    webapp = (
        _spawn_webapp(a.webapp_root, a.port, live_dataset) if a.with_webapp else None
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host, a.port))
    sock.listen(100)
    print(f"[star_server] 监听 {a.host}:{a.port},等待 webapp 连接…", flush=True)
    try:
        while True:
            conn, _addr = sock.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, service, a.mode, active),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        print("\n[star_server] 收到中断,退出。", flush=True)
    finally:
        sock.close()
        if webapp is not None:
            print("[star_server] 关闭 webapp 子进程…", flush=True)
            webapp.terminate()
            try:
                webapp.wait(timeout=5)
            except subprocess.TimeoutExpired:
                webapp.kill()


if __name__ == "__main__":
    main()
