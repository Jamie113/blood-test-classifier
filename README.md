# Blood Test Classifier

A Python tool for discovering patterns in blood test results using unsupervised machine learning. Upload a dataset of real patient results and the app identifies natural clusters — without pre-labelling them — so you can interpret what those clusters mean.

---

## How it works

1. **Upload a CSV** export of blood test results (wide format: one row per patient, one column per marker)
2. **GMM clustering** (Gaussian Mixture Model) is fitted per marker directly from your data. The number of clusters is chosen automatically using BIC scoring
3. **Upload & Discover** shows the discovered clusters for each marker — their means, ranges, and patient counts — alongside reference range lines for context
4. **Population Groups** clusters patients by their full blood test profile (all markers combined), revealing which types of patient exist in your population
5. **Patient View** shows each patient's cluster assignments and flags markers where they fall in the smallest (most unusual) cluster

Clusters are labelled Cluster 1, 2, 3 — not pre-labelled as Normal/Borderline/Abnormal. The goal is to let the data speak first, then you add the meaning.

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

| Blood Test Info Blood Test ID | Blood Test Info Haemoglobin Levels | Blood Test Info HBA1C Levels | ... |
|---|---|---|---|
| 100015 | 163 | 56.83 | ... |

All 27 markers from standard blood panel exports are supported. See `column_map.py` for the full list.

**Unit auto-detection:** Six markers automatically detect and convert between unit systems at upload time:

| Marker | Detection | Conversion |
|---|---|---|
| Testosterone | value > 100 → ng/dL | ÷ 28.84 → nmol/L |
| Free Testosterone | value > 5 → pmol/L | ÷ 1000 → nmol/L |
| Total / LDL / HDL Cholesterol | value > 15 → mg/dL | ÷ 38.67 → mmol/L |
| HbA1C | value < 20 → % | → mmol/mol |
| Oestradiol | value > 200 → pmol/L | ÷ 3.671 → pg/mL |
| Prolactin | value < 50 → ng/mL | × 21.2 → mIU/L |

All values are stored internally in canonical units. The **Unit preferences** panel in the app lets you switch display units instantly without re-running the pipeline.

---

## Reference ranges

All 27 markers have male reference ranges defined in `thresholds.py`. These are shown as dotted context lines on cluster histograms — they are not used to label clusters.

To update a threshold, edit `thresholds.py` — all downstream logic picks it up automatically.

---

## Sample size guidance

Results become meaningful at different thresholds depending on the analysis:

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
app.py              — Streamlit UI (3 tabs: Upload & Discover, Population Groups, Patient View)
gmm.py              — Core GMM functions (fit, sort, boundaries, cluster assignment)
thresholds.py       — Male reference ranges for all 27 markers
column_map.py       — Maps CSV column headers to marker names
unit_conversions.py — Upload-time auto-detection + display unit transforms
tests/              — Tests across all modules
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

---

## Future directions

- Female reference ranges (currently male only)
- Persist uploaded data to SQLite for longitudinal analysis — population clusters improve as more data arrives
- Sequenced blood tests — can the result of one predict the next?
