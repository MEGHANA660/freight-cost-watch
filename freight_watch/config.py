"""All tunable constants in one place. Nothing here is random."""

# --- Baselines (fixed by the case-study brief) -------------------------------
HISTORY_WEEKS = 8  # trailing window for "vs own history", strictly prior weeks

# --- Flag rule ---------------------------------------------------------------
# A route-week is a *candidate* (cost rising AND out of the ordinary) if either:
#   spike: well above own history AND clearly above similar routes
#   drift: modestly above own history AND far above similar routes
#          (catches slow creep that a lagging 8-week average under-reports)
SPIKE_OWN_PCT = 8.0   # above p99 of normal noise (sd 2.8, p99 5.9; evals section 8)
SPIKE_PEER_PCT = 10.0
DRIFT_OWN_PCT = 5.0
DRIFT_PEER_PCT = 20.0

# --- Note grounding ----------------------------------------------------------
NOTE_PROXIMITY_DAYS = 28        # notes this close (but not overlapping) are cited as "closest note"
OPEN_ENDED_HORIZON_WEEKS = HISTORY_WEEKS  # "starting this week" effects stay explanatory until the
                                          # 8-week baseline has absorbed the new level
MAGNITUDE_TOLERANCE_PTS = 4.0   # slack on a note's stated % range (weekly noise)
RETRIEVAL_TOP_K = 3

# --- Output ------------------------------------------------------------------
FLAG_YES = "Yes"
FLAG_JUSTIFIED = "No (justified)"
OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason",
]
