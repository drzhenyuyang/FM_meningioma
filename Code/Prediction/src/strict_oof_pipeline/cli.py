from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the training-only five-fold OOF multimodal MLP pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == "__main__":
    main()
