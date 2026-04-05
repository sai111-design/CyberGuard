"""Pipeline orchestrator — runs all CyberGuard stages in sequence."""

import subprocess
import sys

STEPS = [
    ("python src/data_prep.py",                   "Cleaning NIST dataset"),
    ("python src/mapping/crosswalk_builder.py",    "Building framework crosswalk"),
    ("python src/mapping/control_mapper.py",       "Generating org inventory"),
    ("python src/model/train.py",                  "Training anomaly model"),
    ("python src/model/predict.py",                "Scoring controls & risks"),
    ("python src/detection/alert_generator.py",    "Generating alerts"),
]


def main():
    for i, (cmd, label) in enumerate(STEPS, start=1):
        banner = f"{'='*60}\n  Step {i}/{len(STEPS)}: {label}\n  CMD:  {cmd}\n{'='*60}"
        print(banner)

        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print(f"\nFAILED: {cmd}")
            sys.exit(1)

        print()

    print("=" * 60)
    print("  Pipeline complete. Run: streamlit run src/dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
