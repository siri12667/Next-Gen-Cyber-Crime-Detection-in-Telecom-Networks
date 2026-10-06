"""Run the whole pipeline in the terminal (no GUI needed).

    python -m src.run_cli --data data/sample_telecom_traffic.csv
"""
import argparse
from pathlib import Path

from src.ml_pipeline import CyberCrimePipeline


def main():
    ap = argparse.ArgumentParser(description="Cyber attack detection - CLI")
    ap.add_argument("--data", default="data/sample_telecom_traffic.csv")
    ap.add_argument("--plot", default="outputs/comparison_graph.png")
    args = ap.parse_args()

    p = CyberCrimePipeline()
    print(p.load_dataset(args.data))
    print(p.preprocess())
    print(p.split())
    print(p.train_logistic_regression().summary())
    print(p.train_random_forest().summary())
    print(p.predict_rows(10))
    Path(args.plot).parent.mkdir(exist_ok=True)
    p.comparison_figure().savefig(args.plot, dpi=150)
    print(f"Comparison graph saved -> {args.plot}")
    print(f"Models saved -> {p.save_models()}")


if __name__ == "__main__":
    main()
