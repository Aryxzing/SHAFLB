"""Command-line entry point: `python run_experiment.py [--download]`."""

import argparse

from src.config import RAW_DATA_FILE
from src.data import download_dataset
from src.models import METRIC_LABELS
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SHAFLB federated learning experiment.")
    parser.add_argument("--download", action="store_true", help="download the UCI dataset if it is missing")
    args = parser.parse_args()

    if not RAW_DATA_FILE.exists():
        if not args.download:
            raise SystemExit(f"Dataset missing at {RAW_DATA_FILE}. Re-run with --download.")
        print("Downloading UCI Diabetes 130-US Hospitals dataset...")
        download_dataset()

    results = run_pipeline()
    models = results["models"]
    names = {k: v["label"] for k, v in results["fl_runs"].items()}
    cols = ["centralized_lr", "centralized_rf", *names]
    headers = ["Central LR", "Central RF", *names.values()]

    print(f"\nCohort: {results['dataset']['total_records']:,} patients | "
          f"{results['dataset']['readmission_rate_percent']}% readmitted <30d | "
          f"{results['split']['features']} features")
    print(f"\n{'Metric':<12}" + "".join(f"{h:>13}" for h in headers))
    for key, label in METRIC_LABELS.items():
        print(f"{label:<12}" + "".join(f"{models[c][key]:>13.4f}" for c in cols))
    print(f"\nAudit ledger: {results['ledger']['message']}")


if __name__ == "__main__":
    main()
