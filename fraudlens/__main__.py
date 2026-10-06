"""One command for the whole run (G6): train every model, evaluate, write the report.

python -m fraudlens [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraudlens.config import load_config
from fraudlens.evaluate import evaluate
from fraudlens.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and report.")
    parser.add_argument("--config", default="config.yaml", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    version = train(cfg, args.config.read_text(encoding="utf-8"))
    evaluate(cfg, version)


if __name__ == "__main__":
    main()
