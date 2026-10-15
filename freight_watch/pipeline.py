"""End-to-end run: shipments + notes -> output CSV, weekly metrics, cost log."""
import json
from pathlib import Path

import pandas as pd

from . import config
from .explain import pct, reason_for
from .judge import judge
from .metrics import add_baselines, load_shipments, weekly_route_costs
from .notes import NoteIndex, load_notes


def build_weekly(shipments_path) -> pd.DataFrame:
    return add_baselines(weekly_route_costs(load_shipments(shipments_path)))


def build_rows(weekly: pd.DataFrame, index: NoteIndex) -> pd.DataFrame:
    rows = []
    for _, r in weekly.sort_values(["week_of", "route"]).iterrows():
        v = judge(r, index)
        if v is None:
            continue
        rows.append({
            "route": r["route"],
            "week_of": pd.Timestamp(r["week_of"]).date().isoformat(),
            "cost_per_tonne_km": round(r["cost_per_tonne_km"], 2),
            "vs_own_history": pct(r["vs_own_pct"], "vs this route's past average"),
            "vs_similar_routes": pct(r["vs_peer_pct"], "vs similar-length routes this week"),
            "flagged": config.FLAG_JUSTIFIED if v.justified else config.FLAG_YES,
            "matched_note_id": v.matched.note.note_id if v.justified else "",
            "reason": reason_for(v, r),
        })
    return pd.DataFrame(rows, columns=config.OUTPUT_COLUMNS)


def run(shipments_path, notes_path, out_dir) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weekly = build_weekly(shipments_path)
    index = NoteIndex(load_notes(notes_path))
    result = build_rows(weekly, index)

    result.to_csv(out / "output.csv", index=False)
    weekly.assign(week_of=weekly["week_of"].dt.date).round(6).to_csv(out / "weekly_metrics.csv", index=False)
    # No generative model is called anywhere in the run; log it explicitly.
    (out / "cost_log.json").write_text(json.dumps({
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
        "retrieval": "local TF-IDF vector index (scikit-learn); no API, no embedding model download",
        "note": "Verdicts, citations and reasons are deterministic code over parsed note facts.",
    }, indent=2) + "\n")
    return result
