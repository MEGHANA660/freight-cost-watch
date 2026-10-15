"""Usage: python run.py [--shipments CSV] [--notes CSV] [--out DIR]"""
import argparse
from pathlib import Path

from freight_watch.pipeline import run

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description="Freight cost-creep watcher")
    ap.add_argument("--shipments", default=ROOT / "data/shipment_records.csv")
    ap.add_argument("--notes", default=ROOT / "data/context_notes.csv")
    ap.add_argument("--out", default=ROOT / "outputs")
    a = ap.parse_args()
    res = run(a.shipments, a.notes, a.out)
    print(f"{len(res)} route-weeks reported -> {Path(a.out) / 'output.csv'}")
    print(res["flagged"].value_counts().to_string())


if __name__ == "__main__":
    main()
