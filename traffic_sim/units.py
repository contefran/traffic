"""Unit conversions for the boundary between human-facing km/h and SI m/s.

The simulation core is strictly SI (metres, seconds, m/s, m/s^2) because the
Intelligent Driver Model dynamics are cleanest there and mixing units invites
bugs. Human-facing surfaces — the command line, printed summaries, plot axes —
speak km/h. Convert with these helpers *at the boundary* (on input and for
display) and never inside the core.
"""

# One km/h expressed in m/s: 1000 m per 3600 s.
MS_PER_KMH = 1000.0 / 3600.0


def kmh_to_ms(kmh: float) -> float:
    """Convert a speed from km/h to m/s (e.g. 90 km/h -> 25 m/s)."""
    return kmh * MS_PER_KMH


def ms_to_kmh(ms: float) -> float:
    """Convert a speed from m/s to km/h (e.g. 25 m/s -> 90 km/h)."""
    return ms / MS_PER_KMH
