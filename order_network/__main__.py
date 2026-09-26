import argparse
import importlib

from order_network import config
from order_network.runtime import configure_logging, create_a2a_app, public_url, serve

AGENT_MODULES = {
    "orchestrator": "order_network.agents.orchestrator",
    "order-intake": "order_network.agents.order_intake",
    "inventory": "order_network.agents.inventory",
    "production": "order_network.agents.production",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m order_network", description="Start one component of the A2A network")
    parser.add_argument("component", choices=["registry", *AGENT_MODULES])
    parser.add_argument("--port", type=int, help="port to listen on (default depends on component)")
    args = parser.parse_args()
    port = args.port or config.DEFAULT_PORTS[args.component]
    configure_logging()

    if args.component == "registry":
        from order_network.registry import create_app

        serve(create_app(), port)
        return

    module = importlib.import_module(AGENT_MODULES[args.component])
    app = create_a2a_app(
        module.create_agent(), name=module.NAME, description=module.DESCRIPTION,
        skills=module.SKILLS, url=public_url(port), local_url=f"http://127.0.0.1:{port}",
    )
    serve(app, port)


if __name__ == "__main__":
    main()
