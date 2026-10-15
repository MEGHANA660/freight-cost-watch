"""Ask questions about the findings in plain English (stretch goal).

    python ask.py "why did Chennai-Bangalore get pricier in March?"

No LLM: it recognises the route / month / year in the question, then answers
from outputs/output.csv, quoting the stored reasons (so answers cannot drift
from the audited results).
"""
import calendar
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "outputs/output.csv"
METRICS = Path(__file__).parent / "outputs/weekly_metrics.csv"  # lists ALL routes, flagged or not


def answer(question: str, results: pd.DataFrame, all_routes=None) -> str:
    q = question.lower()
    df = results.fillna("").copy()
    df["dt"] = pd.to_datetime(df["week_of"])
    routes = [r for r in (all_routes if all_routes is not None else df["route"].unique())
              if r.lower() in q or all(p in q for p in r.lower().split("-"))]
    months = [i for i in range(1, 13) if calendar.month_name[i].lower() in q or
              re.search(rf"\b{calendar.month_abbr[i].lower()}\b", q)]
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", q)]
    sel = df
    if routes:
        sel = sel[sel.route.isin(routes)]
    if months:
        sel = sel[sel.dt.dt.month.isin(months)]
    if years:
        sel = sel[sel.dt.dt.year.isin(years)]
    if "unexplained" in q or "flagged" in q:
        sel = sel[sel.flagged == "Yes"]
    scope = " / ".join(routes) or "any route"
    if sel.empty:
        return (f"I found no flagged or justified cost rise for {scope}"
                f"{' in that period' if months or years else ''}. Either costs stayed within the normal "
                "range, or the rise was shared with similar routes (market-wide), which I do not flag.")
    lines = [f"{len(sel)} week(s) for {scope}:"]
    for r in sel.itertuples():
        verdict = (f"JUSTIFIED by {r.matched_note_id}" if r.matched_note_id else "UNEXPLAINED")
        lines.append(f"- {r.route}, week of {r.week_of}: Rs {r.cost_per_tonne_km:.2f}/tonne-km, "
                     f"{r.vs_own_history}, {r.vs_similar_routes}. {verdict}. {r.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python ask.py "your question"')
    print(answer(" ".join(sys.argv[1:]), pd.read_csv(OUT), pd.read_csv(METRICS)["route"].unique()))
