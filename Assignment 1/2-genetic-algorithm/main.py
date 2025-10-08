# main.py
# Runs your GA pipeline scripts in order (no cleanup).
# Expects the numbered files to be in the same folder.

import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Seed check", ["00-gadata.py"]),
    ("Run GA",     ["02-garun.py"]),
]

def run_step(title: str, argv: list[str]) -> None:
    script = ROOT / argv[0]
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    cmd = [sys.executable, str(script), *argv[1:]]
    print(f"\n\033[32m=== {title} ===\033[0m")
    print("All good for now")
    subprocess.run(cmd, check=True)

def main() -> int:
    try:
        for title, argv in STEPS:
            run_step(title, argv)
        print("\nGA pipeline finished.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: step failed with exit code {e.returncode}")
        return e.returncode or 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
