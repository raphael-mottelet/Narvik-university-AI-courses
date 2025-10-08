import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    #("STEP 0: Import and extract main dataset files",                                            ["00-import_openfoodfacts_full.py"]),
    ("STEP 1: Keeping useful columns, THIS IS STEP ONE !",                                       ["01-dropcol.py"]),
    ("STEP 2: Dropping rows of some col who have too much missing values",                       ["02-droprow.py"]),
    ("STEP 3: Droping unwanted food_groups",                                                     ["03-dropfoodcategory.py"]),
    ("STEP 4: Filling default values for missing nutriments value (0.0)",                        ["04-fillmissingdata.py"]),
    ("STEP 5: Dropping data that isnt in english",                                               ["05-droplanguages.py"]),
    ("STEP 6: Droping long ingredients names (mostly not english)",                              ["06-dropfoodingredients.py"]),
    ("STEP 7: Using a synthetic map to transform 2 word ingredients into one",                   ["07-normalize-foodingredients.py"]),
    ("STEP 8: Reducing non-food words to atoms",                                                 ["08-drop-nonrelevantwords.py"]),
    ("STEP 9: adding synthetic data for ingredients price",                                      ["09-add-syntheticprice.py"]),
    ("STEP 10: adding synthetic data for carbon footprint indicator",                            ["10-add-syntheticfootprint.py"]),
    ("STEP 11: Health check to see if the synthetic data is messed up",                          ["11-healthcheck.py"]),
    ("STEP 12: Creating vector for ML ready ",                                                   ["12-make-vectorcore.py"]),
    ("STEP 13: Creating vector for ML ready ",                                                   ["13-make-vectorfull.py"]),
]

OUTPUT_DIR = ROOT.parent / "data" / "openfoodfacts"

KEEP_INTERMEDIATE_STEPS = {10}

def run_pipeline_step(title: str, argv: list[str]) -> None:
    script = ROOT / argv[0]
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    cmd = [sys.executable, str(script), *argv[1:]]
    print(f"\n\033[32m=== {title} ===\033[0m")
    print("->", " ".join(cmd))
    subprocess.run(cmd, check=True)

def cleanup_intermediate_outputs(output_dir: Path, total_steps: int, keep_steps: set | None = None) -> None:
    """
    Delete step1.csv .. step{total_steps-1}.csv in output_dir,
    except those listed in keep_steps. Always keeps the final step{total_steps}.csv.
    Also tolerates 'spte{n}.csv' typos.
    """
    if total_steps <= 1:
        return

    keep_steps = set(keep_steps or set())

    print(f"\n\033[1;36m=== Cleaning the mess: removing step1 to step{total_steps-1}.csv in {output_dir} ===\033[0m")
    for i in range(1, total_steps):
        if i in keep_steps:
            print(f"-> keeping step{i}.csv")
            continue
        for name in (f"step{i}.csv", f"spte{i}.csv"):
            p = output_dir / name
            try:
                if p.exists():
                    p.unlink()
                    print(f"-> deleted {p}")
            except Exception as e:
                print(f"Warning: couldn't delete {p}: {e}", file=sys.stderr)

    final_a = output_dir / f"step{total_steps}.csv"
    final_b = output_dir / f"spte{total_steps}.csv"
    if not final_a.exists() and not final_b.exists():
        print(f"Note: expected final output ({final_a.name} or {final_b.name}) not found.", file=sys.stderr)

def run_pipeline() -> int:
    try:
        for title, argv in STEPS:
            run_pipeline_step(title, argv)

        cleanup_intermediate_outputs(OUTPUT_DIR, len(STEPS), keep_steps=KEEP_INTERMEDIATE_STEPS)

        print("\nPipeline finished successfully.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: step failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(run_pipeline())
