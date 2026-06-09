# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A FastAPI + HTMX app (with a hand-written CSS design system) that discovers natural clusters in blood test results using unsupervised machine learning. Each row in the dataset is one **blood test** (a single panel result), not a patient-as-such. The app does **not** classify results as Normal/Borderline/Abnormal — it finds what groups exist.

Four tabs:
- **1. Groups** — per-marker GMM, histogram + density curves, reference range lines.
- **2. Clusters** — multivariate clustering (PCA → diagonal-covariance GMM), scatter, cluster cards, fingerprint heatmap.
- **3. Outliers** — boundary cases (max posterior < 0.7) + multivariate outliers (χ² test on Mahalanobis distance).
- **4. Correlations** — two-marker scatter coloured by cluster, Pearson r.

A left-rail **cohort filter** (age range + per-marker value ranges) re-runs all four tabs on a filtered subset; filter state lives in URL query params.

A left-rail **upload** form (`POST /upload`) replaces the demo data with a parsed CSV, and a **display units** disclosure overrides the default unit per marker.

## Running the app

```bash
source venv/bin/activate
uvicorn web.main:app --reload
# → http://localhost:8000
```

## Tests

```bash
python -m pytest                          # all tests (236)
python -m pytest tests/test_analysis.py   # core analysis only
```

**236 tests, all should pass.** Run before and after any change to analysis logic.

## Linting

```bash
ruff check .                              # lint the whole tree
```

Ruff is **lint-only** (no repo-wide formatter — the code is hand-aligned).
Config lives in `pyproject.toml`: rule sets `F, E, W, I, C90, B, NPY`, a
`max-complexity = 10` mccabe cap, and per-file `E501` exemptions for the
hand-aligned reference tables. CI (`.github/workflows/tests.yml`) gates every
PR on both `ruff check .` and the pytest suite — keep both green.

## File map

| File / dir | Purpose |
|------------|---------|
| `analysis.py` | Core analysis: `analyse_upload`, `analyse_population`, `build_labelled_df`, `filter_long`, `most_separated_marker`, `strongest_marker_pair` |
| `gmm.py` | Low-level GMM: `fit_optimal_gmm`, `sort_gmm` (returns `(means, stds, weights, order)`), `get_boundaries`, `assign_clusters` |
| `parsing.py` | CSV parser (`parse_csv`) — wide-format upload → long DataFrame. Returns **four** values: `(df_long, recognised, unrecognised, unit_report)` |
| `thresholds.py` | Reference ranges for 27 markers + `classify_test()` |
| `column_map.py` | Maps CSV column headers → marker names |
| `unit_conversions.py` | Upload-time unit auto-detection + display transforms |
| `stub_data.py` | 80 synthetic demo blood tests across two designed sub-populations |
| `demo_cache.pkl` | Pre-baked demo analysis — committed so cold start is instant |
| `bake_demo.py` | Regenerates `demo_cache.pkl` — run after changing `stub_data.py`, `analysis.py`, or `gmm.py` |
| `web/main.py` | FastAPI app: route handlers + the request pipeline (`get_filter_spec`, response builders) |
| `web/contexts.py` | Per-tab template context builders, incl. `_investigate_context` (outlier flagging) |
| `web/charts.py` | Plotly chart helpers (`_marker_chart_html`, `_population_scatter_html`, `_pair_chart_html`, `_heatmap_html`) |
| `web/filters.py` | `FilterSpec` + cohort-filter parsing/normalisation |
| `web/state.py` | `AppState`, demo loading (`_load_demo`), and the `_filtered_data_cached` LRU |
| `web/templates/` | Jinja2 templates: `base.html`, `index.html`, and `partials/*.html` |
| `web/static/styles.css` | Design system: typography scale, colour tokens, components |
| `render.yaml` | Render Blueprint — build command, start command, health check |
| `tests/` | pytest suite |
| `.claude/agents/ml-reviewer.md` | Custom subagent for periodic ML / methodology review (read-only, opus) |

## Key decisions

**`demo_cache.pkl` is committed to git** — intentional. Running all GMM fits on cold start took 60–120s on Render free tier. The pickle loads in ~50ms. `*.pkl` is in `.gitignore` but `demo_cache.pkl` is excepted with `!demo_cache.pkl`.

**Whenever `stub_data.py`, `analysis.py`, or `gmm.py` changes, regenerate the cache:**
```bash
python bake_demo.py
git add demo_cache.pkl
```

**Analysis layer is framework-agnostic** — `analysis.py`, `gmm.py`, `thresholds.py`, `unit_conversions.py`, `parsing.py` and `stub_data.py` have no FastAPI imports. The web layer imports from them; tests import directly.

**`web/main.py` route handlers contain no analysis logic** — all computation lives in `analysis.py` / `gmm.py` / `parsing.py`, context building in `web/contexts.py`, and charts in `web/charts.py`. Routes parse query params, call analysis, and assemble the template context.

### Per-marker GMM (`fit_optimal_gmm` in `gmm.py`)

- Tries **K = 1, 2, … up to `min(4, max(2, n // 5))`** with `n_init=5`, `random_state=42`.
- BIC selection, with a **ΔBIC ≥ 6 floor** against K=1: a K>1 solution only wins if its BIC beats the K=1 fit by 6+ units. This stops uniform markers being forced into spurious sub-groups.
- Hard labels come from `gmm.predict` remapped via `order_inverse` (computed from `sort_gmm`'s order). The boundary-walk is kept only for chart annotations; do not use it to label points.

### Population GMM (`analyse_population` in `analysis.py`)

- **Robust preprocessing for the clustering geometry**: winsorise each marker to its 1st–99th percentile, then `RobustScaler` (median/IQR) → PCA → GMM. Raw `StandardScaler` let a single extreme/erroneous record steer PC1 — on a real upload that produced an unstable split and a one-person "cluster" (116/28/**1**), and the extreme tails also inflated the apparent separation (ΔBIC 166→29 once capped). Do **not** revert to `StandardScaler` for the PCA path.
- **Three deliberate views of the data** (see the comment block in `analyse_population`): standard-scaled *winsorised* values feed only the descriptive fingerprint (z-units the strength copy keys on); winsorised robust-scaled values feed PCA + clustering; the **un-winsorised** projection (`X_full`) feeds only the Mahalanobis distance, so a record whose value was capped for the fit is still flagged as an outlier rather than hidden in a cluster.
- **Well-populated-cluster floor**: K is chosen only among solutions whose smallest component has ≥ `max(3, n_patients // 25)` members (`MIN_CLUSTER_SIZE_FLOOR = 3`, scaled with cohort size to match the K-cap's ~25/cluster logic). A tiny "cluster" is an outlier, not a sub-population. K=1 is always eligible, so the ΔBIC floor still runs.
- **Diagonal covariance** (`covariance_type='diag'`) — PCA already decorrelates globally, and full covariance over-penalises K>1 in small cohorts (~5× more parameters per cluster).
- **K range is sample-size capped**: `range(1, max(2, min(5, n_patients // 25)) + 1)`. n=80 demo can reach at most K=3; K=5 needs ~125 patients.
- **`n_init=5`** (was 3 — bumped after a review found the multivariate fit was occasionally hitting local optima).
- ΔBIC ≥ 6 floor against K=1, same as per-marker.
- Returns `posteriors`, `mahalanobis_sq` (squared distance to assigned cluster's mean — used by Outliers tab), plus the standard fields.

### Outlier flagging (`_investigate_context` in `web/contexts.py`)

- **Boundary cases**: max posterior < 0.7 — split between two clusters.
- **Multivariate outliers**: per-test Mahalanobis² > `chi2.ppf(0.99, df=n_cluster_dims)`. Absolute threshold from the Gaussian model (replaces the old "bottom 5% of log-likelihood" rule, which always flagged ~5% by construction).

### Cohort filter — full vs filtered state

The unfiltered source lives in `state.df_long_full` etc. Filtered analysis is computed lazily by `_filtered_data_cached(spec)` — `functools.lru_cache(maxsize=32)` keeps memory bounded on the 512 MB Render tier (~400 KB per cached entry). Both `_load_demo` and `/upload` call `_filtered_data_cached.cache_clear()`.

Filter state lives in URL query params (`age_min`, `age_max`, repeated `m=Marker:lo:hi`) so cohorts are bookmarkable and tab-switching preserves the filter.

`demo_cache.pkl` covers only the unfiltered case — applying any filter always recomputes.

### `analyse_upload` and `analyse_population` return new fields

If you change the return shape, update both `bake_demo.py` consumers and the tests in `tests/test_analysis.py`.

- **`analyse_upload`** per-marker dict now includes `gmm` (the fitted model) and `order_inverse` (for posterior label remapping). Required for `build_labelled_df` to use `gmm.predict`.
- **`analyse_population`** dict now includes `posteriors` and `mahalanobis_sq` alongside the original fields.

### Design system (`web/static/styles.css`)

Typography scale: `t-eyebrow` (uppercase 0.72rem) / `t-h1` (2.0rem 600 weight) / `t-h2` / `t-body` / `t-meta`. One accent (`--accent: #4C72B0`); 5-step grayscale; cluster colours used only inside chart areas. All chrome (cards, disclosures, toolbar, tab-strip, cohort popover) is custom CSS in `styles.css` — no CSS framework. There is no front-end build step; `styles.css` is served as-is.

Each tab follows the same shape:
1. **Intro card** (boxed) — what the view shows + named ML technique
2. **Methodology disclosure** (collapsed) — expert-level walkthrough of the algorithm, parameters, assumptions
3. **Control bar** (selectors)
4. **Chart / data viz**
5. **Quiet `.finding` block** — single-paragraph summary, less imposing than a headline
6. **Detail sections / disclosures**
7. **Next-step prompt** linking to the next tab

## ML review agent

`.claude/agents/ml-reviewer.md` defines an `ml-reviewer` subagent (read-only, opus). Invoke it to assess methodology, parameter choices, statistical assumptions, and surface improvements. The agent description is wired for proactive triggering when changes touch `analysis.py`, `gmm.py`, `_investigate_context` (`web/contexts.py`), or the chart helpers in `web/charts.py`.

## Workflow for major changes

Anything that spans more than one PR — a new feature, a multi-step refactor, a methodology change — must start with a GitHub milestone and child issues, one per intended PR. Use the `/feature` slash command to scaffold the milestone + issues; do not start writing code until the breakdown is approved by the user.

- One issue = one PR. If an issue grows past that, split it.
- Every PR description must reference its issue with `Closes #N`.
- Trivial single-PR work (typo fixes, dependency bumps, isolated bug fixes) is exempt — go straight to a branch.

## Keeping this file current

Update `CLAUDE.md` whenever the project structure, key decisions, or constraints change — for example:
- A new module is added or an existing one is removed
- A deliberate architectural decision is made (and the reasoning should be preserved)
- A deployment or environment constraint changes
- Something is removed that a future agent might try to re-introduce

Do not update it for routine changes like adding a test or tweaking a threshold value.

## What to avoid

- Do not add analysis logic directly in `web/main.py` route handlers — it belongs in `analysis.py` / `gmm.py` / `parsing.py`.
- Do not remove `demo_cache.pkl` from git or the `.gitignore` exception.
- Do not bypass the ΔBIC ≥ 6 floor when picking K — uniform markers should report K=1.
- Do not switch the population GMM back to `covariance_type='full'` without checking the demo still picks K=2 (full covariance over-penalises K>1 in 10-D space).
- Do not revert the population PCA path to plain `StandardScaler` on un-winsorised values — the winsorise → `RobustScaler` step is what stops one bad record steering the clustering (and the un-winsorised view is kept only for the Mahalanobis outlier distance). Removing it brings back the one-person-"cluster" failure on real uploads.
- Do not return to the 5%-quantile log-likelihood outlier rule — it always flagged ~5% of tests by construction.
- Reference ranges in `thresholds.py` are male-only — do not present them as universal.
- Do not convert upload units **per value** — decide one unit per column via `to_canonical_column` (the per-value `_to_canonical` is private and only feeds conversion-factor tests). Per-value detection lets a single column split across unit systems — the silent-corruption bug fixed in #57. Detection thresholds assume typical adult-male ranges (same caveat as the reference ranges).

## Deployment

Hosted on Render free tier (512 MB RAM). The free tier spins down after 15 min inactivity — 30–60s cold start is unavoidable, but all analysis is instant once Python is running because `demo_cache.pkl` is pre-baked.

`render.yaml` declares the build/start commands and the `/healthz` health-check path. Existing dashboard-created services do not auto-pick-up `render.yaml`; either set the start command to `uvicorn web.main:app --host 0.0.0.0 --port $PORT` manually, or recreate the service from the Blueprint.
