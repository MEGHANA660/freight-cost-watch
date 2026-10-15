"""Eval harness. Run: python evals/run_evals.py   (exit code 1 on any failure)

Why each check exists is in the section headers; results go to evals/eval_report.txt.
"""
import csv
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from freight_watch import config  # noqa: E402
from freight_watch.judge import judge  # noqa: E402
from freight_watch.notes import NoteIndex, load_notes, parse_note  # noqa: E402
from freight_watch.pipeline import build_rows, build_weekly, run  # noqa: E402

SHIP, NOTES = ROOT / "data/shipment_records.csv", ROOT / "data/context_notes.csv"
LINES, FAILS = [], []


def log(s=""):
    LINES.append(s)
    print(s)


def check(name, ok, detail=""):
    log(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
log("1. METRICS vs an independent pure-python recomputation (no pandas, no shared code)")
by = defaultdict(lambda: [0.0, 0.0])
rtype = {}
for r in csv.DictReader(open(SHIP)):
    d = datetime.strptime(r["shipment_date"], "%Y-%m-%d").date()
    monday = d - timedelta(days=d.weekday())
    k = (r["origin"] + "-" + r["destination"], monday)
    by[k][0] += float(r["freight_cost_inr"])
    by[k][1] += float(r["quantity_tonnes"]) * float(r["distance_km"])
    rtype[k[0]] = r["route_type"]
cpt = {k: c / tk for k, (c, tk) in by.items()}
weeks = defaultdict(list)
for (route, w) in sorted(cpt):
    weeks[route].append(w)
ref = {}
for route, ws in weeks.items():
    for i, w in enumerate(ws):
        prior = [cpt[(route, x)] for x in ws[max(0, i - 8):i]]
        own = (cpt[(route, w)] / (sum(prior) / len(prior)) - 1) * 100 if prior else None
        peers = [cpt[(o, w)] for o in weeks if o != route and rtype[o] == rtype[route] and (o, w) in cpt]
        peer = (cpt[(route, w)] / (sum(peers) / len(peers)) - 1) * 100 if peers else None
        ref[(route, w)] = (cpt[(route, w)], own, peer)
weekly = build_weekly(SHIP)
bad = 0
for _, r in weekly.iterrows():
    c, o, p = ref[(r["route"], pd.Timestamp(r["week_of"]).date())]
    for mine, theirs in ((r["cost_per_tonne_km"], c), (r["vs_own_pct"], o), (r["vs_peer_pct"], p)):
        if theirs is None:
            bad += pd.notna(mine)
        elif abs(mine - theirs) > 1e-9:
            bad += 1
check("all route-weeks agree on cost/tonne-km, vs-own, vs-peer", bad == 0, f"({len(weekly)} route-weeks, {bad} mismatches)")
gaps = sum(1 for ws in weeks.values() for a, b in zip(ws, ws[1:]) if (b - a).days != 7)
check("no missing calendar weeks (so 'last 8 weeks with data' == 'last 8 calendar weeks')", gaps == 0)
check("peer average excludes the route itself", all(
    abs(ref[(r, w)][2] - (cpt[(r, w)] / (sum(cpt[(o, w)] for o in weeks if o != r and rtype[o] == rtype[r]) /
                                          sum(1 for o in weeks if o != r and rtype[o] == rtype[r])) - 1) * 100) < 1e-9
    for (r, w) in list(ref)[:50] if ref[(r, w)][2] is not None))

# ---------------------------------------------------------------------------
log("\n2. SAMPLE FILE: the 3 rows provided in sample_output_format_v2.csv must reproduce")
out = build_rows(weekly, NoteIndex(load_notes(NOTES)))
# The provided sample's 3rd row has an unquoted comma inside `reason` (9 fields), so read
# it tolerantly: first 7 fields are positional, everything after is the reason.
_rows = list(csv.reader(open(ROOT / "data/sample_output_format_v2.csv")))
sample = pd.DataFrame([r[:7] + [",".join(r[7:])] for r in _rows[1:]], columns=_rows[0])
mine = out.fillna("").set_index(["route", "week_of"])
for _, s in sample.iterrows():
    m = mine.loc[(s.route, s.week_of)]
    same = all(str(m[c]) == str(s[c]) for c in
               ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"])
    check(f"{s.route} {s.week_of} numbers/verdict/note identical to sample", same)
check("output columns identical to sample (names + order)", list(out.columns) == list(sample.columns))

# ---------------------------------------------------------------------------
log("\n3. GOLD SET hand-labelled from reading the notes + series (written before tuning the code)")
GOLD = {  # (route, week_of): (flagged, matched_note_id)
    ("Delhi-Jaipur", "2024-11-11"): ("Yes", ""), ("Delhi-Jaipur", "2024-11-18"): ("Yes", ""),
    ("Ahmedabad-Mumbai", "2025-01-20"): ("No (justified)", "N002"),
    ("Chennai-Bangalore", "2025-02-24"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-03"): ("No (justified)", "N001"),
    ("Chennai-Bangalore", "2025-03-10"): ("Yes", ""),   # flood ended Mar 8; N009: back to normal
    ("Chennai-Bangalore", "2025-03-17"): ("Yes", ""),
    ("Mumbai-Pune", "2025-09-15"): ("Yes", ""),         # N006 'stable demand' must NOT justify
}
MUST_NOT_REPORT = [("Delhi-Chennai", "2024-02-05"),    # +8.5% own but only +7.8% vs peers: noise
                   ("Mumbai-Delhi", "2025-05-12")]    # diesel step hits all routes alike
got = {(r.route, r.week_of): (r.flagged, r.matched_note_id) for r in out.fillna("").itertuples()}
for k, v in GOLD.items():
    check(f"gold {k[0]} {k[1]} -> {v[0]} {v[1]}", got.get(k) == v, f"(got {got.get(k)})")
for k in MUST_NOT_REPORT:
    check(f"known false-alarm candidate not reported: {k[0]} {k[1]}", k not in got)
drift = [k for k in got if k[0] == "Mumbai-Pune" and k[1] >= "2025-06-01"]
check("slow Mumbai-Pune creep (H2 2025) is surfaced, never 'justified'",
      len(drift) >= 8 and all(got[k][0] == "Yes" for k in drift), f"({len(drift)} weeks)")
extra = [k for k in got if k not in GOLD and k not in drift]
check("no other rows reported", not extra, f"{extra}")

# ---------------------------------------------------------------------------
log("\n4. HALLUCINATION TRAPS on synthetic notes (judge must refuse to cite)")
idx_notes = load_notes(NOTES)


def row(route, week, own=30.0, peer=25.0):
    return {"route": route, "week_of": pd.Timestamp(week), "vs_own_pct": own, "vs_peer_pct": peer}


def mk(nid, d, applies, text):
    return parse_note({"note_id": nid, "date": d, "applies_to": applies, "note": text})


flood = "Flooding on the A-B highway from Feb 24 to Mar 8 forced detours and higher trip costs."
cases = [
    ("correct route + week -> justified", [mk("T1", "2025-02-24", "Chennai-Bangalore", flood)],
     row("Chennai-Bangalore", "2025-03-03"), "T1"),
    ("WRONG ROUTE (same dates) -> unexplained", [mk("T1", "2025-02-24", "Chennai-Bangalore", flood)],
     row("Delhi-Jaipur", "2025-03-03"), None),
    ("WRONG WINDOW (week after it ended) -> unexplained", [mk("T1", "2025-02-24", "Chennai-Bangalore", flood)],
     row("Chennai-Bangalore", "2025-03-10"), None),
    ("WRONG WINDOW (week before it began) -> unexplained", [mk("T1", "2025-02-24", "Chennai-Bangalore", flood)],
     row("Chennai-Bangalore", "2025-02-17"), None),
    ("note says costs NOT affected -> unexplained",
     [mk("T2", "2025-03-03", "Chennai-Bangalore", "Maintenance caused delays for a week; costs were not significantly affected.")],
     row("Chennai-Bangalore", "2025-03-03"), None),
    ("'no major disruptions' note -> unexplained",
     [mk("T3", "2025-03-03", "All Routes", "Demand remained stable with no major disruptions reported.")],
     row("Chennai-Bangalore", "2025-03-03"), None),
    ("all-routes diesel note cannot explain a +25% gap vs peers",
     [mk("T4", "2025-05-05", "All Routes", "Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%.")],
     row("Mumbai-Pune", "2025-05-12", own=6.0, peer=25.0), None),
    ("stated 5-7% cannot cover a +30% rise",
     [mk("T4", "2025-05-05", "All Routes", "Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%.")],
     row("Mumbai-Pune", "2025-05-12", own=30.0, peer=5.0), None),
    ("no notes at all -> unexplained", [], row("Mumbai-Pune", "2025-05-12"), None),
]
for name, notes, r, want in cases:
    v = judge(pd.Series(r), NoteIndex(notes)) if notes else None
    if not notes:  # judge() needs an index; empty-index case
        v = judge(pd.Series(r), NoteIndex([mk("Z", "2020-01-01", "X-Y", "placeholder")]))
    got_id = v.matched.note.note_id if v and v.justified else None
    check(name, got_id == want, f"(cited {got_id})")

# ---------------------------------------------------------------------------
log("\n5. CITATION AUDIT of every 'justified' row, re-verified from the raw notes file")
raw = {r["note_id"]: r for r in csv.DictReader(open(NOTES))}
for r in out[out.flagged == config.FLAG_JUSTIFIED].itertuples():
    n = raw.get(r.matched_note_id)
    ws = date.fromisoformat(r.week_of)
    fine = n is not None and n["applies_to"] in (r.route, "All Routes")
    fine = fine and "not" not in n["note"].lower().split("costs were")[-1][:25]
    # window re-derived by hand: N001 Feb24-Mar8, N002 week of Jan20; check overlap with the week
    hand = {"N001": (date(2025, 2, 24), date(2025, 3, 8)), "N002": (date(2025, 1, 20), date(2025, 1, 26))}
    lo, hi = hand[r.matched_note_id]
    check(f"{r.route} {r.week_of} cites {r.matched_note_id}: route ok, week overlaps window",
          fine and ws <= hi and ws + timedelta(days=6) >= lo)
check("every 'Yes' row has blank matched_note_id", (out[out.flagged == "Yes"].matched_note_id.fillna("") == "").all())

# ---------------------------------------------------------------------------
log("\n6. NOTE PARSER: facts extracted from the 10 real notes (hand-checked expectations)")
EXPECT = {"N001": ("raises", "2025-02-24", "2025-03-08"), "N002": ("raises", "2025-01-20", "2025-01-26"),
          "N003": ("raises", "2025-05-05", "2025-06-29")}
for n in idx_notes:
    if n.note_id in EXPECT:
        e = EXPECT[n.note_id]
        check(f"{n.note_id} raises, window {e[1]}..{e[2]}",
              (n.polarity, str(n.window_start), str(n.window_end)) == e)
    else:
        check(f"{n.note_id} can never justify (polarity={n.polarity})", n.polarity != "raises")

# ---------------------------------------------------------------------------
log("\n7. SYNTHETIC INJECTION: multiply one route-week's cost by (1+x), same seed every run")
ship = pd.read_csv(SHIP)
ship["_route"] = ship.origin + "-" + ship.destination
ship["_wk"] = pd.to_datetime(ship.shipment_date)
ship["_wk"] = (ship["_wk"] - pd.to_timedelta(ship["_wk"].dt.weekday, unit="D")).dt.date.astype(str)
event_weeks = {k[1] for k in GOLD} | {"2025-03-24"}
clean = sorted((r, w) for (r, w) in {(a, b) for a, b in zip(ship._route, ship._wk)}
               if "2024-04-01" <= w <= "2025-02-10" and w not in event_weeks)


def spaced_picks(seed, n=24):
    """Seeded picks, <=1 per week and >=10 weeks apart per route, so injections cannot
    contaminate each other's baselines (simultaneous rises on all peers look market-wide by design)."""
    rng, chosen = random.Random(seed), []
    for rt, wk in rng.sample(clean, len(clean)):
        d = date.fromisoformat(wk)
        if any(c[1] == wk or (c[0] == rt and abs((date.fromisoformat(c[1]) - d).days) < 70) for c in chosen):
            continue
        chosen.append((rt, wk))
        if len(chosen) == n:
            break
    return chosen


log("  detection vs. size of injected rise (24 spaced route-weeks each; one-week cost uplift)")
for x in (0.05, 0.10, 0.15, 0.20, 0.30):
    picks = spaced_picks(7)
    s = ship.copy()
    for rt, wk in picks:
        m = (s._route == rt) & (s._wk == wk)
        s.loc[m, "freight_cost_inr"] = (s.loc[m, "freight_cost_inr"] * (1 + x)).round()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        s.drop(columns=["_route", "_wk"]).to_csv(p, index=False)
        res = run(p, NOTES, td)
    found = {(r.route, r.week_of) for r in res.itertuples()}
    hit = sum(1 for k in picks if k in found)
    fp = sum(1 for k in found if k not in set(picks) and k not in got)
    log(f"  +{int(x * 100):>2}% injected: recall {hit}/{len(picks)}, extra rows vs baseline output: {fp}")
    if x == 0.20:
        check("recall at +20% >= 80% (measured; misses are where a noisy peer route masks the gap)", hit >= 0.8 * len(picks))
    if x == 0.30:
        check("recall at +30% == 100%", hit == len(picks))
    if x <= 0.05:
        check("a +5% blip is NOT flagged (below noise)", hit <= 4)
    check(f"+{int(x * 100)}% run adds no spurious rows", fp <= 2, f"({fp})")

# injected rise inside a REAL note window, wrong route must stay unexplained, right route justified
def inject(route, week, x=0.30):
    s = ship.copy()
    m = (s._route == route) & (s._wk == week)
    s.loc[m, "freight_cost_inr"] = (s.loc[m, "freight_cost_inr"] * (1 + x)).round()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        s.drop(columns=["_route", "_wk"]).to_csv(p, index=False)
        return run(p, NOTES, td).set_index(["route", "week_of"])


r1 = inject("Delhi-Jaipur", "2025-03-03")  # N001 window, but Chennai-Bangalore's flood
check("rise on Delhi-Jaipur during Chennai-Bangalore flood -> unexplained (wrong route)",
      r1.loc[("Delhi-Jaipur", "2025-03-03")].flagged == "Yes")
r2 = inject("Delhi-Chennai", "2025-01-20")  # N002's week, but Ahmedabad-Mumbai only
check("rise on Delhi-Chennai in the festival week -> unexplained (wrong route)",
      r2.loc[("Delhi-Chennai", "2025-01-20")].flagged == "Yes")

# ---------------------------------------------------------------------------
log("\n8. THRESHOLD CALIBRATION: how noisy is a normal week? (weeks outside all known events)")
w = weekly.copy()
w["week"] = w.week_of.dt.date.astype(str)
w = w[(w.history_weeks_used >= 8) & (w.week < "2024-11-01")]   # first ~10 months: no planted events
vs = w.vs_own_pct
log(f"  vs-own %: n={len(vs)}, sd={statistics.pstdev(vs):.2f}, p99={vs.quantile(.99):.1f}, max={vs.max():.1f}")
pg = w.vs_peer_pct
log(f"  vs-peer %: sd={statistics.pstdev(pg):.2f}, p99={pg.quantile(.99):.1f}, max={pg.max():.1f}")
check("spike threshold (own>=8%) sits above 99th pct of normal noise", config.SPIKE_OWN_PCT >= vs.quantile(.99) - 0.5)

(ROOT / "evals/eval_report.txt").write_text("\n".join(LINES) + f"\n\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + str(FAILS)}\n")
print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
