"""Microgrid_AI - command-line entry point.

Project: *Generative AI vs Agentic AI for Microgrid Operation: A Comparative
Analysis*. Each subcommand maps to a build stage; stages are filled in
incrementally. Run `python main.py --help` to see all commands.
"""
from __future__ import annotations

import argparse

from utils.logger import get_logger

log = get_logger("main")

_STUB = "[Stage {n}] '{cmd}' is scaffolded but not yet implemented."


def cmd_data(args: argparse.Namespace) -> None:
    # Imported lazily so `--help` and other stages don't pay the pandas import.
    from datagen.generate import generate_dataset

    generate_dataset(days=args.days, seed=args.seed)


def cmd_forecast(_args: argparse.Namespace) -> None:
    from forecast.train import train_all

    train_all()


def cmd_baseline(_args: argparse.Namespace) -> None:
    from ems.baseline import RuleBasedController
    from ems.runner import run_controller, save_run

    sim, metrics = run_controller(RuleBasedController())
    save_run(sim, metrics)


def cmd_llm(args: argparse.Namespace) -> None:
    from ems.environment import load_eval_window
    from ems.runner import run_controller, save_run
    from llm.generative import GenerativeController

    window = load_eval_window()
    if args.hours:
        window = window.head(args.hours)
        log.info("Limiting run to first %d hour(s) for testing.", args.hours)
    sim, metrics = run_controller(GenerativeController(), window)
    save_run(sim, metrics)


def cmd_agentic(args: argparse.Namespace) -> None:
    from agents.agentic import AgenticController
    from ems.environment import load_eval_window
    from ems.runner import run_controller, save_run

    window = load_eval_window()
    if args.hours:
        window = window.head(args.hours)
        log.info("Limiting run to first %d hour(s) for testing.", args.hours)
    sim, metrics = run_controller(AgenticController(), window)
    save_run(sim, metrics)


def cmd_compare(args: argparse.Namespace) -> None:
    from backend.compare import run_comparison

    run_comparison(rerun=args.rerun)


def cmd_dashboard(args: argparse.Namespace) -> None:
    import uvicorn

    log.info("Starting dashboard at http://%s:%d  (Ctrl+C to stop)",
             args.host, args.port)
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)


def cmd_site(_args: argparse.Namespace) -> None:
    from frontend.build_site import build_site

    build_site()


def cmd_report(args: argparse.Namespace) -> None:
    from report.generate import generate_report

    generate_report(formats=args.format)


# (command name, help text, handler)
_COMMANDS = [
    ("data", "Stage 1: build & feature-engineer the dataset", cmd_data),
    ("forecast", "Stage 2: train forecasting models (solar/wind/demand/SoC)", cmd_forecast),
    ("baseline", "Stage 3: run the rule-based EMS baseline", cmd_baseline),
    ("llm", "Stage 4: run the Generative-AI (LangChain + Groq) approach", cmd_llm),
    ("agentic", "Stage 5: run the Agentic-AI (LangGraph) approach", cmd_agentic),
    ("compare", "Stage 6: run the comparison harness (baseline vs LLM vs agentic)", cmd_compare),
    ("dashboard", "Stage 7: launch the FastAPI dashboard", cmd_dashboard),
    ("site", "Stage 8: build the static explanation website", cmd_site),
    ("report", "Stage 9: generate the report (Markdown / HTML / Word)", cmd_report),
]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with one subcommand per stage."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Generative AI vs Agentic AI for Microgrid Operation - staged build CLI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run a stage, e.g.:  python main.py data",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers = {}
    for name, help_text, func in _COMMANDS:
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(func=func)
        subparsers[name] = sp

    # Stage 1 options.
    subparsers["data"].add_argument(
        "--days", type=int, default=365,
        help="days of hourly data to generate (default: 365)",
    )
    subparsers["data"].add_argument(
        "--seed", type=int, default=42,
        help="random seed for reproducibility (default: 42)",
    )

    # Stage 4/5 options: cap hours for a quick/cheap test run.
    for _name in ("llm", "agentic"):
        subparsers[_name].add_argument(
            "--hours", type=int, default=None,
            help="only run the first N hours of the eval window (for testing)",
        )

    # Stage 6 option: force fresh runs instead of reusing saved ones.
    subparsers["compare"].add_argument(
        "--rerun", action="store_true",
        help="re-run all three approaches instead of reusing saved runs",
    )

    # Stage 9 option: pick one or more output formats (default: all).
    subparsers["report"].add_argument(
        "--format", nargs="+", choices=["md", "html", "docx"], default=None,
        metavar="FMT",
        help="report formats to generate: md html docx (default: all three)",
    )

    # Stage 7 options: dashboard host/port.
    subparsers["dashboard"].add_argument(
        "--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)",
    )
    subparsers["dashboard"].add_argument(
        "--port", type=int, default=8000, help="bind port (default: 8000)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
