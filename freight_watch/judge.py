"""Flag rule + grounding check. Pure, deterministic functions."""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import pandas as pd

from . import config
from .notes import Note, NoteIndex


@dataclass
class Assessment:
    note: Note
    ok: bool
    code: str            # OK | NOT_RAISING | WINDOW_ENDED | WINDOW_NOT_STARTED | SYSTEMIC | TOO_SMALL
    score: float = 0.0


@dataclass
class Verdict:
    trigger: str                           # "spike" | "drift"
    justified: bool
    matched: Optional[Assessment]          # the citing note when justified
    nearby: list = field(default_factory=list)  # non-justifying assessments, closest first


def trigger_for(row) -> Optional[str]:
    """Return 'spike'/'drift' if the week is rising AND out of the ordinary, else None."""
    own, peer = row["vs_own_pct"], row["vs_peer_pct"]
    if pd.isna(own) or pd.isna(peer):
        return None  # no history or no peers -> cannot establish "unusual"; never guess
    if own >= config.SPIKE_OWN_PCT and peer >= config.SPIKE_PEER_PCT:
        return "spike"
    if own >= config.DRIFT_OWN_PCT and peer >= config.DRIFT_PEER_PCT:
        return "drift"
    return None


def _week_span(week_of):
    start = pd.Timestamp(week_of).date()
    return start, start + timedelta(days=6)


def assess(note: Note, row, score: float = 0.0) -> Assessment:
    """Can this note justify THIS route-week's rise? Every condition must hold."""
    ws, we = _week_span(row["week_of"])
    if note.polarity != "raises":
        return Assessment(note, False, "NOT_RAISING", score)
    if note.window_end < ws:
        return Assessment(note, False, "WINDOW_ENDED", score)
    if note.window_start > we:
        return Assessment(note, False, "WINDOW_NOT_STARTED", score)
    if note.scope == "all" and row["vs_peer_pct"] >= config.SPIKE_PEER_PCT:
        # A market-wide cause lifts similar routes too; it cannot explain a gap vs them.
        return Assessment(note, False, "SYSTEMIC", score)
    if note.magnitude_hi is not None and row["vs_own_pct"] > note.magnitude_hi + config.MAGNITUDE_TOLERANCE_PTS:
        return Assessment(note, False, "TOO_SMALL", score)
    return Assessment(note, True, "OK", score)


def _gap_days(note: Note, row) -> int:
    ws, we = _week_span(row["week_of"])
    return max(0, (note.window_start - we).days, (ws - note.window_end).days)


def judge(row, index: NoteIndex) -> Optional[Verdict]:
    trig = trigger_for(row)
    if trig is None:
        return None
    ranked = index.retrieve(row["route"], k=10_000)   # scope-filtered, best match first
    assessed = [assess(n, row, s) for n, s in ranked if _gap_days(n, row) <= config.NOTE_PROXIMITY_DAYS]
    ok = [a for a in assessed if a.ok]
    if ok:  # route-specific evidence beats an all-routes note; then retrieval score; then id
        best = sorted(ok, key=lambda a: (a.note.scope == "all", -a.score, a.note.note_id))[0]
        return Verdict(trig, True, best)
    return Verdict(trig, False, None, nearby=assessed)
