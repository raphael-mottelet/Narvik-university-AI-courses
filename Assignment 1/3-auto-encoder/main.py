# main.py
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [("Train autoencoder", ["autoencodertest.py"])]

def run_step(title: str, script_name: str) -> None:
    script_path = ROOT / script_name

    print(f"\n\033[32m=== {title} ===\033[0m")
    subprocess.run([sys.executable, str(script_path)], check=True)

def main() -> int:
    try:
        for title, argv in STEPS:
            run_step(title, argv[0])
        print("\nPipeline finished.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: step failed with exit code {e.returncode}")
        return e.returncode or 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
