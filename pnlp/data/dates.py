"""Shared date utilities for quarterly rebalancing schedules."""

from __future__ import annotations

from datetime import date

__all__ = ["generate_quarterly_dates"]


def generate_quarterly_dates(start: date, end: date) -> list[date]:
    """Generate quarter-start dates between start and end.

    Returns dates on the first day of each calendar quarter
    (Jan 1, Apr 1, Jul 1, Oct 1) that fall within [start, end].
    """
    dates = []
    month = ((start.month - 1) // 3) * 3 + 1
    current = date(start.year, month, 1)
    while current <= end:
        if current >= start:
            dates.append(current)
        month = current.month + 3
        year = current.year
        if month > 12:
            month -= 12
            year += 1
        current = date(year, month, 1)
    return dates
