#!/usr/bin/env python3
import json, subprocess, time, os, sys, requests
from pathlib import Path

def run(cmd, log_path=None, env=None, bg=False):
    print(f"[CMD] {' '.join(cmd)}")
    if bg:
        fh = open(log_path, "w") if log_path else subprocess.DEVNULL
        return subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)
    else:
        with open(log_path, "w") if log_path else subprocess.DEVNULL as fh:
            subprocess.check_call(cmd, stdout=fh, stderr=fh, env=env)

def wait_health(url, timeout=600, interval=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url).ok:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

def start_monitor(cfg_path):
    monitor_dir = Path("/workspace/monitor_logs")
    monitor_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = Path(cfg_path).stem
    monitor_log = monitor_dir / f"{cfg_name}.csv"
    monitor_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,"
        "memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
        "--format=csv",
        "--loop-ms=1000"
    ]
    return run(monitor_cmd, str(monitor_log), env=None, bg=True)

def stop_monitor(proc):
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)

    paths = cfg["paths"]
    for key in ["server_log", "client_log", "result_json"]:
        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1"
    })

    server_cmd = cfg["server_command"]
    client_cmd = cfg["client_command"]
    port = server_cmd[-1]
    health_url = f"http://127.0.0.1:{port}/health"

    monitor_proc = start_monitor(cfg_path)
    server_proc = None
    try:
        server_proc = run(server_cmd, paths["server_log"], env, bg=True)

        print("等待服务就绪...")
        if not wait_health(health_url):
            server_proc.terminate()
            raise RuntimeError("服务启动超时，查看日志: " + paths["server_log"])
        print("服务已就绪")

        env["VLLM_BENCH_RESULT_PATH"] = paths["result_json"]
        run(client_cmd, paths["client_log"], env)
        print("测试完成，结果文件:", paths["result_json"])
    finally:
        if server_proc:
            server_proc.terminate()
        stop_monitor(monitor_proc)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: run_vllm_from_json.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
