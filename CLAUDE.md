# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

An app that discovers natural clusters in blood test results using unsupervised machine learning. Each row in the dataset is one **blood test** (a single panel result), not a patient-as-such. The app does **not** classify results as Normal/Borderline/Abnormal — it finds what groups exist.

There are **two front-ends** sharing the same analysis layer:

- **FastAPI + HTMX + Tailwind** under `web/` — the current development target. Notion-style design system, four tabs, full design control.
- **Streamlit** at the project root (`app.py`) — kept working for comparison; not where new UI features go.

Four tabs (FastAPI app):
- **1. Groups** — per-marker GMM, histogram + density curves, reference range lines.
- **2. Clusters** — multivariate clustering (PCA → diagonal-covariance GMM), scatter, cluster cards, fingerprint heatmap.
- **3. Outliers** — boundary cases (max posterior < 0.7) + multivariate outliers (χ² test on Mahalanobis distance).
- **4. Correlations** — two-marker scatter coloured by cluster, Pearson r.

A left-rail **cohort filter** (age range + per-marker value ranges) re-runs all four tabs on a filtered subset; filter state lives in URL query params.

A left-rail **upload** form (`POST /upload`) replaces the demo data with a parsed CSV, and a **display units** disclosure overrides the default unit per marker.

## Running the apps

```bash
source venv/bin/activate

# FastAPI app (current target)
uvicorn web.main:app --reload

# Streamlit app (legacy, still functional)
streamlit run app.py
```

## Tests

```bash
python -m pytest                          # all tests (191)
python -m pytest tests/test_analysis.py   # core analysis only
```

**191 tests, all should pass.** Run before and after any change to analysis logic.

## File map

| File / dir | Purpose |
|------------|---------|
| `analysis.py` | Core analysis: `analyse_upload`, `analyse_population`, `build_labelled_df`, `filter_long`, `most_separated_marker`, `strongest_marker_pair` |
| `gmm.py` | Low-level GMM: `fit_optimal_gmm`, `sort_gmm` (returns `(means, stds, weights, order)`), `get_boundaries`, `assign_clusters` |
| `parsing.py` | Shared CSV parser (`parse_csv`) — used by both FastAPI and Streamlit apps |
| `thresholds.py` | Reference ranges for 27 markers + `classify_test()` |
| `column_map.py` | Maps CSV column headers → marker names |
| `unit_conversions.py` | Upload-time unit auto-detection + display transforms |
| `stub_data.py` | 80 synthetic demo blood tests across two designed sub-populations |
| `demo_cache.pkl` | Pre-baked demo analysis — committed so cold start is instant |
| `bake_demo.py` | Regenerates `demo_cache.pkl` — run after changing `stub_data.py`, `analysis.py`, or `gmm.py` |
| `app.py` | Streamlit UI (legacy) |
| `web/main.py` | FastAPI app: routes, context builders, chart helpers, request pipeline |
| `web/templates/` | Jinja2 templates: `base.html`, `index.html`, and `partials/*.html` |
| `web/static/styles.css` | Design system: typography scale, colour tokens, components |
| `tests/` | pytest suite |
| `.claude/agents/ml-reviewer.md` | Custom subagent for periodic ML / methodology review (read-only, opus) |

## Key decisions

**`demo_cache.pkl` is committed to git** — intentional. Running all GMM fits on cold start took 60–120s on Render free tier. The pickle loads in ~50ms. `*.pkl` is in `.gitignore` but `demo_cache.pkl` is excepted with `!demo_cache.pkl`.

**Whenever `stub_data.py`, `analysis.py`, or `gmm.py` changes, regenerate the cache:**
```bash
python bake_demo.py
git add demo_cache.pkl
```

**Analysis layer is framework-agnostic** — `analysis.py`, `gmm.py`, `thresholds.py`, `unit_conversions.py`, `parsing.py` and `stub_data.py` have no Streamlit or FastAPI imports. Both UIs import from them.

**Streamlit `app.py` contains no analysis logic** — all computation lives in `analysis.py`. `app.py` only reads from `st.session_state` and renders.

**No `@st.cache_data` on analysis functions** — DataFrame hashing overhead exceeded the benefit. Session state is the cache; analysis only runs when a new file is uploaded.

### Per-marker GMM (`fit_optimal_gmm` in `gmm.py`)

- Tries **K = 1, 2, … up to `min(4, max(2, n // 5))`** with `n_init=5`, `random_state=42`.
- BIC selection, with a **ΔBIC ≥ 6 floor** against K=1: a K>1 solution only wins if its BIC beats the K=1 fit by 6+ units. This stops uniform markers being forced into spurious sub-groups.
- Hard labels come from `gmm.predict` remapped via `order_inverse` (computed from `sort_gmm`'s order). The boundary-walk is kept only for chart annotations; do not use it to label points.

### Population GMM (`analyse_population` in `analysis.py`)

- **Diagonal covariance** (`covariance_type='diag'`) — PCA already decorrelates globally, and full covariance over-penalises K>1 in small cohorts (~5× more parameters per cluster).
- **K range is sample-size capped**: `range(1, max(2, min(5, n_patients // 25)) + 1)`. n=80 demo can reach at most K=3; K=5 needs ~125 patients.
- **`n_init=5`** (was 3 — bumped after a review found the multivariate fit was occasionally hitting local optima).
- ΔBIC ≥ 6 floor against K=1, same as per-marker.
- Returns `posteriors`, `log_likelihood`, `mahalanobis_sq` (squared distance to assigned cluster's mean — used by Outliers tab), `z_scores`, plus the standard fields.

### Outlier flagging (`_investigate_context` in `web/main.py`)

- **Boundary cases**: max posterior < 0.7 — split between two clusters.
- **Multivariate outliers**: per-test Mahalanobis² > `chi2.ppf(0.99, df=n_cluster_dims)`. Absolute threshold from the Gaussian model (replaces the old "bottom 5% of log-likelihood" rule, which always flagged ~5% by construction).

### Cohort filter — full vs filtered state

**Streamlit** keeps two parallel sets of analysis results in `st.session_state`: `*_full` (unfiltered source, populated once on upload or demo load) and the active set (filtered subset). Re-runs on filter change.

**FastAPI** stores only the unfiltered source in `state.df_long_full` etc.; filtered analysis is computed by `_filtered_data_cached(spec)` (LRU cache, max 32 entries to bound memory on the 512 MB tier). `_load_demo` and `/upload` both call `_filtered_data_cached.cache_clear()`.

Filter state lives in URL query params (`age_min`, `age_max`, repeated `m=Marker:lo:hi`) so cohorts are bookmarkable and tab-switching preserves the filter.

`demo_cache.pkl` covers only the unfiltered case — applying any filter always recomputes.

### `analyse_upload` and `analyse_population` return new fields

If you change the return shape, update both `bake_demo.py` consumers and the tests in `tests/test_analysis.py`.

- **`analyse_upload`** per-marker dict now includes `gmm` (the fitted model) and `order_inverse` (for posterior label remapping). Required for `build_labelled_df` to use `gmm.predict`.
- **`analyse_population`** dict now includes `posteriors`, `log_likelihood`, `mahalanobis_sq`, and `z_scores` alongside the original fields.

### FastAPI design system (`web/static/styles.css`)

Typography scale: `t-eyebrow` (uppercase 0.72rem) / `t-h1` (2.0rem 600 weight) / `t-h2` / `t-body` / `t-meta`. One accent (`--accent: #4C72B0`); 5-step grayscale; cluster colours used only inside chart areas. Most chrome (cards, disclosures, control-bar, tab-strip) is custom CSS; Tailwind is loaded via CDN for utility classes only.

Each tab follows the same shape:
1. **Intro card** (boxed) — what the view shows + named ML technique
2. **Methodology disclosure** (collapsed) — expert-level walkthrough of the algorithm, parameters, assumptions
3. **Control bar** (selectors)
4. **Chart / data viz**
5. **Quiet `.finding` block** — single-paragraph summary, less imposing than a headline
6. **Detail sections / disclosures**
7. **Next-step prompt** linking to the next tab

## ML review agent

`.claude/agents/ml-reviewer.md` defines an `ml-reviewer` subagent (read-only, opus). Invoke it to assess methodology, parameter choices, statistical assumptions, and surface improvements. The agent description is wired for proactive triggering when changes touch `analysis.py`, `gmm.py`, `_investigate_context`, or chart helpers.

## Keeping this file current

Update `CLAUDE.md` whenever the project structure, key decisions, or constraints change — for example:
- A new module is added or an existing one is removed
- A deliberate architectural decision is made (and the reasoning should be preserved)
- A deployment or environment constraint changes
- Something is removed that a future agent might try to re-introduce

Do not update it for routine changes like adding a test or tweaking a threshold value.

## What to avoid

- Do not add analysis logic directly in `app.py` or `web/main.py` route handlers — it belongs in `analysis.py` / `gmm.py` / `parsing.py`.
- Do not add `@st.cache_data` to functions that take DataFrames as arguments.
- Do not remove `demo_cache.pkl` from git or the `.gitignore` exception.
- Do not bypass the ΔBIC ≥ 6 floor when picking K — uniform markers should report K=1.
- Do not switch the population GMM back to `covariance_type='full'` without checking the demo still picks K=2 (full covariance over-penalises K>1 in 10-D space).
- Do not return to the 5%-quantile log-likelihood outlier rule — it always flagged ~5% of tests by construction.
- Reference ranges in `thresholds.py` are male-only — do not present them as universal.

## Deployment

Hosted on Render free tier (512 MB RAM). The free tier spins down after 15 min inactivity — 30–60s cold start is unavoidable, but all analysis is instant once Python is running because `demo_cache.pkl` is pre-baked. The Streamlit deployment uses `streamlit run app.py`; the FastAPI deployment is `uvicorn web.main:app`.
