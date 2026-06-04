from pathlib import Path

import subprocess
import pandas as pd


def resolve_input_path(user_path: Path | None) -> Path:
    return 'test.tsv'


def main() -> None:
    subprocess.run(['python3', 'predict.py'])


if __name__ == "__main__":
    main()
