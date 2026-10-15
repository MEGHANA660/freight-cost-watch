"""Context notes: structured facts, a vector index, and retrieval.

Each note is parsed ONCE into explicit facts (scope, polarity, effect window,
stated magnitude). Retrieval ranks notes; the facts, not free text, decide
whether a note can justify a cost rise. Anything the parser is unsure about
resolves to the conservative option ("unclear"/"no_effect"), which can never
justify a flag.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

# Notes saying costs were NOT pushed up (checked first: conservative).
NO_EFFECT_CUES = [
    "not significantly affected", "no significant disruption", "no major disruption",
    "without a rate change", "returned to normal", "remained normal", "remained stable",
    "not part of this dataset", "improved road conditions", "absorbed",
]
RAISES_CUES = ["higher trip costs", "surcharge", "pushing up", "prices rose", "costs rise", "higher costs"]
RANGE_RE = re.compile(r"from ([A-Z][a-z]{2}) (\d{1,2}) to ([A-Z][a-z]{2}) (\d{1,2})")
PCT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    applies_to: str
    text: str
    scope: str                 # "route" | "all"
    polarity: str              # "raises" | "no_effect" | "unclear"
    window_start: date
    window_end: date
    magnitude_hi: Optional[float]  # upper end of stated % rise, if any

    def first_sentence(self) -> str:
        return re.split(r"(?<=\.)\s", self.text)[0].rstrip(".")


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def parse_note(row) -> Note:
    nd = pd.Timestamp(row["date"]).date()
    text, applies = str(row["note"]), str(row["applies_to"]).strip()
    low = text.lower()

    if any(c in low for c in NO_EFFECT_CUES):
        polarity = "no_effect"
    elif any(c in low for c in RAISES_CUES):
        polarity = "raises"
    else:
        polarity = "unclear"

    m = RANGE_RE.search(text)
    if m:  # explicit "from Feb 24 to Mar 8"
        start = date(nd.year, MONTHS[m[1]], int(m[2]))
        end = date(nd.year, MONTHS[m[3]], int(m[4]))
    elif "starting this week" in low or "starting from this week" in low:  # open-ended step change
        start = _monday(nd)
        end = start + timedelta(weeks=config.OPEN_ENDED_HORIZON_WEEKS) - timedelta(days=1)
    else:  # point event / "festival week": the note's own Mon-Sun week
        start = _monday(nd)
        end = start + timedelta(days=6)

    pm = PCT_RANGE_RE.search(text)
    return Note(
        note_id=str(row["note_id"]), date=nd, applies_to=applies, text=text,
        scope="all" if applies.lower() == "all routes" else "route",
        polarity=polarity, window_start=start, window_end=end,
        magnitude_hi=float(pm[2]) if pm else None,
    )


def load_notes(path) -> list:
    df = pd.read_csv(path)
    return [parse_note(r) for _, r in df.iterrows()]


class NoteIndex:
    """TF-IDF vector index over notes (deterministic, offline, no model download).

    The embedder is deliberately isolated here so a sentence-embedding model can
    replace it without touching the judging logic.
    """

    def __init__(self, notes: list):
        self.notes = sorted(notes, key=lambda n: n.note_id)
        docs = [f"{n.applies_to} {n.text}" for n in self.notes]
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vec.fit_transform(docs)

    def retrieve(self, route: str, k: int = config.RETRIEVAL_TOP_K) -> list:
        """Top-k notes that apply to this route (or all routes), best match first.

        Scope filtering happens BEFORE ranking so a note about another route can
        never be returned. Ties break on note_id for determinism.
        """
        query = (f"{route.replace('-', ' ')} {route} freight cost rise increase surcharge "
                 "diesel toll flooding weather festival disruption")
        sims = (self._matrix @ self._vec.transform([query]).T).toarray().ravel()
        ranked = sorted(
            ((s, n) for s, n in zip(sims, self.notes)
             if n.scope == "all" or n.applies_to.lower() == route.lower()),
            key=lambda t: (-round(float(t[0]), 12), t[1].note_id))
        return [(n, float(s)) for s, n in ranked[:k]]
