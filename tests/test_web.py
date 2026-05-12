"""Web-layer smoke tests. Verifies the FastAPI app boots and the methodology
drawer is wired correctly on every tab."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.main import app

client = TestClient(app)


@pytest.mark.parametrize("tab", ["explorer", "population", "investigate", "pairs"])
def test_each_tab_renders_drawer_trigger_with_correct_section(tab: str) -> None:
    res = client.get(f"/?tab={tab}")
    assert res.status_code == 200
    html = res.text
    assert 'class="methodology-trigger"' in html
    assert f'data-mdr-section="{tab}"' in html
    assert 'id="methodology-drawer"' in html


def test_drawer_contains_all_four_sections() -> None:
    res = client.get("/")
    html = res.text
    for anchor in ("methodology-gmm", "methodology-clusters",
                   "methodology-outliers", "methodology-correlations"):
        assert f'id="{anchor}"' in html


def test_methodology_param_is_passed_through_harmlessly() -> None:
    """Unknown query params are ignored server-side; the drawer opens via
    client-side JS using URLSearchParams. This test asserts the route does not
    500 on the param and still renders the trigger."""
    res = client.get("/?tab=explorer&methodology=explorer")
    assert res.status_code == 200
    assert 'class="methodology-trigger"' in res.text


def test_tab_partial_includes_trigger() -> None:
    """HTMX tab swaps return page_body.html, which must include the trigger so
    the data-mdr-section attribute always matches the now-active tab."""
    res = client.get("/tab/population")
    assert res.status_code == 200
    assert 'data-mdr-section="population"' in res.text


def test_old_methodology_disclosure_is_gone() -> None:
    res = client.get("/")
    html = res.text
    assert "disclosure methodology" not in html
    assert "how this view is computed" not in html.lower()
