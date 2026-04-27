# Blood Test Classifier

A Streamlit app for discovering patterns in blood test results using unsupervised machine learning. Upload a dataset of patient results and the app identifies natural clusters — without pre-labelling them — so you can interpret what those clusters mean.

---

## How it works

1. **Upload a CSV** export of blood test results (wide format: one row per patient, one column per marker). Without a CSV the app loads 80 synthetic demo patients.
2. **GMM clustering** (Gaussian Mixture Model) is fitted per marker directly from your data. The number of clusters is chosen automatically using BIC scoring.
3. **Blood marker explorer** (tab 1) shows the discovered clusters for each marker — their means, ranges, and patient counts — alongside reference range lines for context.
4. **Patient population** (tab 2) clusters patients by their full blood test profile (all markers combined), revealing which patient types exist in your population and what distinguishes them.

Clusters are labelled Group 1, 2, 3 — not pre-labelled as Normal/Borderline/Abnormal. The goal is to let the data speak first, then you add the meaning.

---

## Running the app

```bash
# First time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the app
streamlit run app.py
```

---

## CSV format

The app expects a wide-format export with one row per patient. Column names must match those in `column_map.py`. Example columns:

| Blood Test Info Blood Test ID | Current Age | Blood Test Info Haemoglobin Levels | Blood Test Info HBA1C Levels | ... |
|---|---|---|---|---|
| 100015 | 42 | 163 | 56.83 | ... |

All 27 markers from standard blood panel exports are supported. `Current Age` is optional — if present it enables age-based colouring on the patient scatter plot.

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

All values are stored internally in canonical units. The **⚙ Units** button (top-right of the tab bar) lets you switch display units instantly without re-running the analysis.

---

## Reference ranges

All 27 markers have male reference ranges defined in `thresholds.py`. These are shown as dotted context lines on cluster histograms — they are not used to label clusters.

To update a threshold, edit `thresholds.py` — all downstream logic picks it up automatically.

---

## Sample size guidance

| Patients | What's reliable |
|---|---|
| < 30 | Illustrative only |
| 30–100 | Per-marker clustering for well-separated distributions |
| 100–200 | Per-marker clustering reliable; population clustering noisy |
| 200–500 | Population grouping meaningful |
| 500+ | Strong confidence in both |

---

## Project structure

```
app.py              — Streamlit UI (two tabs: marker explorer + population view)
analysis.py         — Core analysis functions (GMM per marker, population clustering)
gmm.py              — Low-level GMM functions (fit, sort, boundaries, assignment)
thresholds.py       — Male reference ranges for all 27 markers + classify_test()
column_map.py       — Maps CSV column headers to marker names
unit_conversions.py — Upload-time unit auto-detection + display unit transforms
stub_data.py        — Generates 80 synthetic demo patients (fixed seed)
demo_cache.pkl      — Pre-baked demo analysis results (loaded instantly on cold start)
bake_demo.py        — Script to regenerate demo_cache.pkl after data/analysis changes
tests/              — Tests across all modules
```

---

## Updating the demo cache

`demo_cache.pkl` is committed to the repo so Render loads it instantly on cold start instead of running all GMM fits. Regenerate it whenever `stub_data.py` or `analysis.py` changes:

```bash
python3 bake_demo.py
git add demo_cache.pkl
git commit -m "Regenerate demo cache"
```

---

## Tests

```bash
python -m pytest
```

| File | Coverage |
|---|---|
| `test_thresholds.py` | All 27 markers, boundary cases, error handling |
| `test_gmm_functions.py` | fit, sort, boundaries, cluster assignment |
| `test_unit_conversions.py` | All 6 conversion rules, boundary detection |
| `test_column_map.py` | All 27 export columns mapped to valid thresholds |
| `test_analysis.py` | analyse_upload, analyse_population, build_labelled_df |

---

## Future directions

- Female reference ranges (currently male only)
- Persist uploaded data for longitudinal analysis — population clusters improve as more data arrives
- Sequenced blood tests — can the result of one predict the next?
