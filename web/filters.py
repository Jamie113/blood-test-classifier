"""Cohort-filter data structures and tab resolution.

Pure module — no FastAPI / state imports. Defines the URL-serialisable
filter spec used by every route that consumes the cohort filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

VALID_TABS = {"explorer", "population", "investigate", "pairs"}


@dataclass(frozen=True)
class MarkerFilter:
    marker: str
    lo: float
    hi: float


@dataclass(frozen=True)
class FilterSpec:
    age_min: int | None = None
    age_max: int | None = None
    markers: tuple = ()  # tuple[MarkerFilter, ...]

    def is_active(self) -> bool:
        return (
            self.age_min is not None
            or self.age_max is not None
            or len(self.markers) > 0
        )

    def cache_key(self) -> tuple:
        return (self.age_min, self.age_max, self.markers)

    @classmethod
    def from_request(
        cls,
        age_min: int | None,
        age_max: int | None,
        m: list[str] | None,
    ) -> "FilterSpec":
        markers: list[MarkerFilter] = []
        for s in (m or []):
            try:
                name, lo, hi = s.split(":", 2)
                markers.append(MarkerFilter(name.strip(), float(lo), float(hi)))
            except (ValueError, IndexError):
                continue
        markers.sort(key=lambda mf: mf.marker)
        return cls(age_min=age_min, age_max=age_max, markers=tuple(markers))

    def to_query_params(self) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        if self.age_min is not None:
            params.append(("age_min", str(self.age_min)))
        if self.age_max is not None:
            params.append(("age_max", str(self.age_max)))
        for mf in self.markers:
            params.append(("m", f"{mf.marker}:{mf.lo}:{mf.hi}"))
        return params

    def to_query_string(self) -> str:
        return urlencode(self.to_query_params())

    def without_marker(self, marker: str) -> "FilterSpec":
        return FilterSpec(
            age_min=self.age_min,
            age_max=self.age_max,
            markers=tuple(mf for mf in self.markers if mf.marker != marker),
        )


def _resolve_tab(tab: str) -> str:
    return tab if tab in VALID_TABS else "explorer"
