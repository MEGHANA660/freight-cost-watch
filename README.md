# Freight Cost-Creep Watcher

Spots route-weeks where freight cost per tonne-km is rising in a way that looks unusual, then
either **cites the exact note that explains it** or says it is **unexplained**. Nothing is guessed.

```bash
pip install pandas scikit-learn
python run.py                      # -> outputs/output.csv (+ weekly_metrics.csv, cost_log.json)
python evals/run_evals.py          # 54 checks, exit code 1 on any failure                 
python ask.py "why did Chennai-Bangalore get pricier in March?"   # stretch: Q&A over results
```

## How it works

| Stage | File | What it does |
|---|---|---|
| 1. Weekly metrics | `metrics.py` | Group by route + `route_type`; Monday-Sunday weeks (`week_of` = Monday); `cost_per_tonne_km = sum(cost) / sum(tonnes x km)` for the route-week. |
| 2. Baselines | `metrics.py` | **vs own history**: mean of the route's prior 8 weekly values, current week excluded (no look-ahead). **vs similar routes**: mean of all *other* routes with the same `route_type` in the same week. Both exactly as the brief defines. |
| 3. Candidate rule | `judge.py` | Rising *and* out of the ordinary: **spike** = >=+8% vs own history AND >=+10% vs peers; **drift** = >=+5% vs own AND >=+20% vs peers (catches slow creep that a lagging average under-reports). Thresholds live in `config.py`. |
| 4. Note retrieval | `notes.py` | Each note is parsed once into facts: scope (route / all routes), polarity (raises / no effect / unclear), effect window, stated % range. A TF-IDF vector index ranks the notes that apply to the route. |
| 5. Grounding | `judge.py` | A note justifies a row only if **all** hold: it raises costs; it applies to this route (or all routes); its window overlaps this week; if it is market-wide it cannot explain a gap vs peers; its stated % range can cover the rise. Otherwise: unexplained. |
| 6. Reason text | `explain.py` | Templates over the note's own words, so the note id, dates and wording are never invented. |

Output has one row per candidate week (17 rows: 3 justified, 14 unexplained), matching the sample columns.

### Decisions worth knowing
- **Flag threshold calibration.** On the first ~10 months (no planted events) week-to-week noise vs own history is sd 2.8%, p99 5.9%, max 6.6% (`eval_report.txt` sec. 8). 8% sits above that. The peer gate (+10%) is what separates a route-specific problem from a market-wide move: the nationwide diesel step in May 2025 raises every route ~6%, so it correctly produces no flags.
- **Open-ended notes** ("starting this week", N003) explain rises only for 8 weeks, by which time the 8-week baseline has absorbed the new level.
- **Flood note N001** (Feb 24 - Mar 8): weeks of 02-24 and 03-03 are justified; 03-10 and 03-17 are still expensive but the flood had ended (N001) and N009 says conditions were normal, so they are **unexplained**.
- **Mumbai-Pune H2-2025** creeps up to +39% over peers with no supporting note. N006 ("stable demand, no major disruptions") is shown as the closest note but is *not* cited. This is the "quiet creep" case; it appears as 10 drift/spike rows.
- **History <8 weeks:** the first week of each route has no baseline (never flagged); weeks 2-8 use all prior weeks available, as the brief says. `weekly_metrics.csv` has `history_weeks_used`. No reported row relies on fewer than 8.

## How I convinced myself it is right (`evals/run_evals.py`, 54 checks, all pass)
1. **Independent recomputation**: metrics re-implemented in pure Python with no shared code; all 728 route-weeks match to 1e-9.
2. **Sample rows**: the three rows in `sample_output_format_v2.csv` reproduce (numbers, verdict, note id).
3. **Hand-labelled gold set** (8 rows written from the notes, plus two "must not report" noise/diesel weeks).
4. **Hallucination traps** on synthetic notes: wrong route, week after/before the window, "costs not affected" notes, market-wide note vs a peer gap, a 5-7% note vs a +30% rise, no notes: all must be unexplained.
5. **Citation audit**: every justified row re-checked against the raw notes (route, window overlap).
6. **Injection test** (seeded, spaced route-weeks): detection of an injected one-week cost uplift:

   | uplift | +5% | +10% | +15% | +20% | +30% |
   |---|---|---|---|---|---|
   | recalled (of 24) | 2 | 11 | 18 | 21 | 24 |

   Zero spurious rows at every size, and a rise injected during another route's flood or festival week stays unexplained.

## Reproducibility
`repro_check.sh` runs three separate processes and diffs every output file: identical (sha256 in `outputs/repro_check.txt`). Nothing is random, no LLM is called, retrieval ties break on note id, and the check also passes with a randomised `PYTHONHASHSEED`.

## Cost of running (one full run)
**0 LLM calls, 0 input tokens, 0 output tokens, $0.** (`outputs/cost_log.json`). Retrieval is a local TF-IDF index.

## Honest limitations
- **No generative model.** I chose deterministic code over an LLM because the guardrails (cite exactly, no fabrication, identical reruns) are easiest to prove that way. The retrieval layer is a TF-IDF vector index, not a semantic embedding model (model downloads were blocked in my sandbox); with only 10 notes this changes nothing, but it would at scale. `NoteIndex` is the one place to swap in embeddings, and `explain.py` is the one place an LLM could paraphrase (temperature 0, verdict/note/numbers stay code-decided).
- **Note parsing is rule-based** (regex + cue phrases), validated on these 10 notes. A new note phrased differently resolves to "unclear", which can never justify, so errors fail safe (unexplained) rather than fabricate.
- **Simultaneous rises on all peer routes look market-wide** and are not flagged; likewise a route-specific spike can be partly masked if a peer spikes the same week (peer average includes it, per the brief's definition).
- **Thresholds are judgment calls**, calibrated on noise and the injection curve above; they are in one file.
- The provided sample CSV has an unquoted comma in its 3rd `reason` (9 fields); my writer quotes properly, so the file parses in any CSV reader.
- Only candidate weeks are output (as in the sample), not all 728 route-weeks; those are in `weekly_metrics.csv`.
