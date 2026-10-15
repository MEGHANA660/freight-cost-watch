"""Plain-English reasons. Templates over verbatim note text: nothing is invented.

An LLM could paraphrase these, but every fact a reason states (note id, date,
window, wording) comes straight from the parsed note, so the verdict, cited
note and numbers never depend on a model.
"""
from .judge import Assessment, Verdict


def pct(x: float, suffix: str) -> str:
    return f"{x:+.1f}% {suffix}"


def _why_not(a: Assessment, row) -> str:
    n = a.note
    if a.code == "NOT_RAISING":
        return (f'says "{n.first_sentence()}" -- it does not describe a reason for a cost rise on this route')
    if a.code == "WINDOW_ENDED":
        return (f"describes a cost-raising event covering {n.window_start} to {n.window_end}, "
                "which ended before this week -- it cannot explain a rise now")
    if a.code == "WINDOW_NOT_STARTED":
        return f"only starts on {n.window_start}, after this week -- it cannot explain this rise"
    if a.code == "SYSTEMIC":
        return (f"describes a market-wide rise (roughly up to {n.magnitude_hi:g}%) that hit similar routes too -- "
                f"it cannot explain being {row['vs_peer_pct']:+.1f}% above them")
    if a.code == "TOO_SMALL":
        return (f"describes a rise of only up to about {n.magnitude_hi:g}%, far below the "
                f"{row['vs_own_pct']:+.1f}% seen here")
    return "does not apply"


def reason_for(v: Verdict, row) -> str:
    if v.justified:
        n = v.matched.note
        return (f'Matches note {n.note_id} dated {n.date}: "{n.text}" '
                f"This note applies to {n.applies_to.lower() if n.scope == 'all' else n.applies_to} "
                f"and covers {n.window_start} to {n.window_end}, which includes this week. "
                "The cost rise has a clear explanation.")
    if not v.nearby:
        return ("No matching note found for this route or date range. "
                "Cost rise looks unexplained and worth a human review.")
    parts = []
    for i, a in enumerate(v.nearby[:2]):
        lead = "The closest note" if i == 0 else "The next closest note"
        parts.append(f"{lead} ({a.note.note_id}, {a.note.date}) {_why_not(a, row)}.")
    return " ".join(parts) + " No genuine justification found; flagged for review."
