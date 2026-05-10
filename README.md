# Blood Test Classifier

A FastAPI + HTMX + Tailwind app for discovering patterns in blood test results using unsupervised machine learning. Upload a wide-format CSV of blood test results and the app identifies natural clusters — without pre-labelling them — so you can interpret what those clusters mean.

Each row in the dataset is treated as one **blood test** (a single panel of marker results), not a patient-as-such.

---

## How it works

1. **Upload a CSV** export of blood test results (wide format: one row per test, one column per marker). Without an upload the app loads 80 synthetic demo blood tests across two designed sub-populations.
2. **Per-marker GMM** (Gaussian Mixture Model) is fitted to each marker's distribution. The number of components is chosen by BIC, with a ΔBIC ≥ 6 floor against K = 1 so uniform markers are honestly reported as having a single group.
3. **Multivariate clustering** runs a Principal Component Analysis on the standardised wide table, then fits a diagonal-covariance GMM in the reduced space. K is sample-size capped (`max(2, min(5, n // 25))`) and again subject to the ΔBIC ≥ 6 floor.
4. **Outlier flagging** uses two rules from the cluster model: tests whose maximum cluster posterior is below 0.7 (boundary cases) and tests whose squared Mahalanobis distance to their assigned cluster's mean exceeds the χ² critical value at p = 0.01 (multivariate outliers).
5. **Pearson correlation** between every pair of markers; the strongest pair is auto-selected as the starting view.

Cluster labels are abstract numbers — Cluster 1, 2, 3 — not pre-labelled as Normal/Borderline/Abnormal. The data speaks first; you add the meaning.

---

## Running the app

```bash
# First time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the app — http://localhost:8000
uvicorn web.main:app --reload
```

---

## The four tabs

1. **Groups** — per-marker GMM. For each marker, shows whether tests fall into one consistent range or split into distinct groups. Auto-picks the marker with the clearest separation.
2. **Clusters** — multivariate clustering. Treats every test as a point in marker space and groups tests with similar overall profiles. Includes a fingerprint heatmap of which markers most distinguish each cluster.
3. **Outliers** — boundary cases (split between two clusters) and multivariate outliers (fit no cluster well). Each row carries the patient ID, age, assigned cluster, confidence percentage, and the reason for flagging.
4. **Correlations** — pairwise marker correlations. Auto-picks the strongest |r| pair; X/Y selectors swap into any other pair.

A left-rail **cohort filter** (age range + per-marker value ranges) re-runs all four tabs on a filtered subset. Filter state lives in URL query params, so cohorts are bookmarkable and tab-switching preserves them.

---

## CSV format

Wide-format export with one row per blood test. Column names must match those in `column_map.py`. Example:

| Blood Test Info Blood Test ID | Current Age | Blood Test Info Haemoglobin Levels | Blood Test Info HBA1C Levels | ... |
|---|---|---|---|---|
| 100015 | 42 | 163 | 56.83 | ... |

All 27 markers from standard blood panel exports are supported. `Current Age` is optional — if present it enables age-based colouring on the cluster scatter plot.

See `column_map.py` for the full column list.

**Unit auto-detection:** Six markers automatically detect and convert between unit systems at upload time:

| Marker | Detection | Conversion |
|---|---|---|
| Testosterone | value > 100 → ng/dL | ÷ 28.84 → nmol/L |
| Free Testosterone | value > 5 → pmol/L | ÷ 1000 → nmol/L |
| Total / LDL / HDL Cholesterol | value > 15 → mg/dL | ÷ 38.67 → mmol/L |
| HbA1C | value < 20 → % | → mmol/mol |
| Oestradiol | value > 200 → pmol/L | ÷ 3.671 → pg/mL |
| Prolactin | value < 50 → ng/mL | × 21.2 → mIU/L |

All values are stored internally in canonical units. The **Display units** section in the left rail lets you override the display unit per marker without re-running the analysis.

---

## Reference ranges

All 27 markers have **male reference ranges** defined in `thresholds.py`. These are shown as dotted context lines on cluster histograms and pair scatters — they are not used to label clusters or to fit any model.

To update a threshold, edit `thresholds.py` — all downstream logic picks it up automatically.

---

## Sample size guidance

| Tests | What's reliable |
|---|---|
| < 30 | Illustrative only |
| 30–100 | Per-marker clustering for well-separated distributions |
| 100–200 | Per-marker clustering reliable; multivariate clusters noisy (K capped at 4 by sample-size rule) |
| 200–500 | Multivariate clustering meaningful; K can reach 5 |
| 500+ | Strong confidence in both |

---

## Project structure

```
analysis.py         — Core analysis (per-marker GMM, multivariate clustering, filter, ranking helpers)
gmm.py              — Low-level GMM (fit_optimal_gmm, sort_gmm, get_boundaries, assign_clusters)
parsing.py          — CSV parser (parse_csv): wide-format upload → long DataFrame
thresholds.py       — Male reference ranges for all 27 markers + classify_test()
column_map.py       — Maps CSV column headers to marker names
unit_conversions.py — Unit auto-detection + display unit transforms
stub_data.py        — 80 synthetic demo blood tests (fixed seed, two designed sub-populations)
demo_cache.pkl      — Pre-baked demo analysis (committed for instant cold start)
bake_demo.py        — Regenerates demo_cache.pkl after data/analysis changes
web/
  main.py           — FastAPI app: routes, context builders, chart helpers
  templates/        — Jinja2 templates (base + partials per tab)
  static/styles.css — Design system: typography, colour, components
render.yaml         — Render Blueprint (build/start commands, /healthz check)
tests/              — pytest suite (191 tests across all modules)
.claude/agents/     — Custom subagents (ml-reviewer for periodic methodology review)
```

---

## Updating the demo cache

`demo_cache.pkl` is committed so Render loads it instantly on cold start instead of running all GMM fits. Regenerate it whenever `stub_data.py`, `analysis.py`, or `gmm.py` changes:

```bash
python3 bake_demo.py
git add demo_cache.pkl
git commit -m "Regenerate demo cache"
```

---

## Tests

```bash
python -m pytest                          # 191 tests across all modules
python -m pytest tests/test_analysis.py   # core analysis only
```

| File | Coverage |
|---|---|
| `test_thresholds.py` | All 27 markers, boundary cases, error handling |
| `test_gmm_functions.py` | fit_optimal_gmm (incl. K=1 floor, ΔBIC margin, sample-size cap), sort_gmm (incl. order remap), boundaries, cluster assignment |
| `test_unit_conversions.py` | All 6 conversion rules, boundary detection |
| `test_column_map.py` | All 27 export columns mapped to valid thresholds |
| `test_analysis.py` | analyse_upload, analyse_population (incl. K=1 + Mahalanobis fields), build_labelled_df, filter_long, most_separated_marker, strongest_marker_pair |

---

## Deployment

Hosted on Render free tier (512 MB RAM). The free tier spins down after 15 minutes of inactivity — 30–60s cold start is unavoidable, but all analysis is instant once Python is running because `demo_cache.pkl` is pre-baked.

Render config lives in `render.yaml`:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn web.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/healthz`
- Python: 3.13.2 (also pinned in `runtime.txt`)

For an existing Render service that wasn't created from a Blueprint, update the start command in the dashboard manually (Render does not auto-pick-up `render.yaml` for services already created).

---

## Future directions

- Female reference ranges (currently male only).
- Persist uploaded data for longitudinal analysis — clusters improve as more data arrives.
- Sequenced blood tests — can the result of one predict the next?
- Spearman correlation default in the Correlations tab (Pearson assumes linearity and is outlier-sensitive).
- Replace BIC with ICL or a parallel-analysis-based dimensionality choice for the multivariate cluster model.
