"""Start the registry and all agents as separate processes on separate ports (Ctrl+C stops everything).

With REGISTRY_BACKEND=litellm no own registry is started; the agents register in the LiteLLM proxy instead.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import httpx

from order_network import a2a_client, config

AGENTS = ["order-intake", "inventory", "production", "orchestrator"]
COLORS = {"registry": 90, "orchestrator": 35, "order-intake": 36, "inventory": 32, "production": 33, "client": 34}


def pump(name: str, stream, log_file, lock: threading.Lock) -> None:
    for line in iter(stream.readline, ""):
        prefix = f"[{name:<12}]"
        with lock:
            print(f"\033[{COLORS[name]}m{prefix}\033[0m {line}", end="", flush=True)
            log_file.write(f"{prefix} {line}")
            log_file.flush()


def start(name: str, args: list[str], env: dict, log_file, lock) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, *args], env=env, cwd=config.PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    threading.Thread(target=pump, args=(name, process.stdout, log_file, lock), daemon=True).start()
    return process


def wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {what}")


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGINT, _raise_interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the demo client once the network is up, then stop")
    parser.add_argument("--agent-host", default="host.docker.internal" if config.REGISTRY_BACKEND == "litellm" else "localhost",
                        help="host name under which agents register (default: host.docker.internal for a LiteLLM "
                             "proxy running in Docker, localhost otherwise)")
    args = parser.parse_args()

    log_path = config.PROJECT_ROOT / "logs" / f"network_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    base_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(config.PROJECT_ROOT)}
    processes: list[subprocess.Popen] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            litellm = config.REGISTRY_BACKEND == "litellm"
            registry = None
            if litellm:
                wait_for(lambda: httpx.get(f"{config.REGISTRY_URL}/health/liveliness").is_success, 20, "LiteLLM proxy")
            else:
                registry = start("registry", ["-m", "order_network", "registry"], base_env, log_file, lock)
                processes.append(registry)
                wait_for(lambda: httpx.get(f"{config.REGISTRY_URL}/health").is_success, 20, "registry")

            agents = {}
            for name in AGENTS:
                port = config.DEFAULT_PORTS[name]
                env = {**base_env, "AGENT_PUBLIC_URL": f"http://{args.agent_host}:{port}"}
                agents[name] = start(name, ["-m", "order_network", name], env, log_file, lock)
                processes.append(agents[name])

            wait_for(lambda: len(asyncio.run(a2a_client.discover())) >= len(AGENTS), 60, "all agents to register")
            where = f"LiteLLM agent registry at {config.REGISTRY_URL}" if litellm else "registry"
            print(f"\n✅ Network up: {len(AGENTS)} agents registered in {where}. Combined log: {log_path.name}\n", flush=True)

            if args.demo:
                client = start("client", ["client.py"], base_env, log_file, lock)
                client.wait()
            else:
                print("Press Ctrl+C to stop. In another terminal run: python client.py\n", flush=True)
                reported = set()
                while registry is None or registry.poll() is None:
                    for name, process in agents.items():
                        if name not in reported and process.poll() is not None:
                            reported.add(name)
                            print(f"⚠ {name} exited with code {process.returncode}.", flush=True)
                    time.sleep(1)
                print("Registry exited, shutting down.", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
