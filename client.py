"""Client: discovers the order fulfilment agent in the registry and sends it a customer order via A2A."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime

import httpx

from order_network import a2a_client, config

DEMO_ORDER = (
    "Hello, this is Anna Berger from Alpine Robotics GmbH. We would like to order 50 hydraulic pump housings "
    "(material 149449), twelve flange couplings 252654 and 100 pieces of 255565. Our purchase order reference is "
    "9938812 and our customer number is 90417. We need the complete delivery by 2026-10-30. Thanks!"
)


class Output:
    def __init__(self):
        path = config.PROJECT_ROOT / "logs" / f"client_{datetime.now():%Y%m%d_%H%M%S}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self._file.write(text + "\n")

    def close(self) -> None:
        self._file.close()


def print_agents(out: Output, agents: list[dict]) -> None:
    out(f"Registry {config.REGISTRY_URL} lists {len(agents)} agent(s):")
    for agent in agents:
        skills = ", ".join(s["id"] for s in agent["card"].get("skills", []))
        out(f"  - {agent['name']:<32} {agent['url']:<28} skills: {skills}")


async def run(skill: str, text: str, list_only: bool) -> int:
    out = Output()
    try:
        agents = await a2a_client.discover()
        print_agents(out, agents)
        if list_only:
            return 0

        out(f"\n🔎 Looking up an agent with skill '{skill}'...")
        target = await a2a_client.resolve(skill)
        out(f"→ {target['name']} at {target['url']}")
        out(f"\n👤 Request:\n{text}\n")

        started = time.monotonic()
        result = await a2a_client.send_task(
            target["url"], text, agent_name=target["name"],
            on_state=lambda name, state: out(f"   [{time.monotonic() - started:6.1f}s] task state: {state}"),
        )
        out(f"\n--- Result from {result.agent} (task {result.task_id}) ---")
        out(json.dumps(result.data, indent=2, ensure_ascii=False) if result.data is not None else result.text)
        return 0
    except httpx.ConnectError:
        out(f"❌ Registry not reachable at {config.REGISTRY_URL}. Start the network first (python run_local.py).")
        return 1
    except a2a_client.A2AError as e:
        out(f"❌ {e}")
        return 1
    finally:
        out(f"\nClient log written to {out.path.relative_to(config.PROJECT_ROOT)}")
        out.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a customer order to the A2A agent network")
    parser.add_argument("text", nargs="?", default=DEMO_ORDER, help="customer order text (default: demo order)")
    parser.add_argument("--skill", default="order-fulfilment", help="skill to call (default: order-fulfilment)")
    parser.add_argument("--list", action="store_true", help="only list registered agents")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.skill, args.text, args.list)))


if __name__ == "__main__":
    main()
