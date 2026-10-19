"""Run the full experiment suite (same as `python -m nashgrid --demo`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nashgrid import cli

if __name__ == "__main__":
    sys.exit(cli.main(["--demo"]))
