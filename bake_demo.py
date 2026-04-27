#!/usr/bin/env python3
"""
Pre-bake demo analysis results into demo_cache.pkl.

Run once locally after any change to stub_data.py or analysis.py,
then commit demo_cache.pkl so Render loads it instantly on cold start.

Usage:
    python bake_demo.py
"""
import pickle
import time
from pathlib import Path

from stub_data import generate_stub_data
from analysis import analyse_upload, analyse_population, build_labelled_df

out = Path(__file__).parent / "demo_cache.pkl"

print("Generating demo data…")
t0 = time.perf_counter()
df_long = generate_stub_data()
print(f"  {len(df_long['patient_id'].unique())} patients, {len(df_long['test_name'].unique())} markers  ({time.perf_counter()-t0:.1f}s)")

print("Fitting per-marker GMMs…")
t1 = time.perf_counter()
gmm_results = analyse_upload(df_long)
print(f"  {len(gmm_results)} markers fitted  ({time.perf_counter()-t1:.1f}s)")

print("Running population clustering…")
t2 = time.perf_counter()
pop_results = analyse_population(df_long)
print(f"  {pop_results.get('n_clusters', '?')} clusters found  ({time.perf_counter()-t2:.1f}s)")

print("Building labelled dataframe…")
df_labelled = build_labelled_df(df_long, gmm_results)

cache = {
    "df_long":     df_long,
    "gmm_results": gmm_results,
    "pop_results": pop_results,
    "df_labelled": df_labelled,
}

with open(out, "wb") as f:
    pickle.dump(cache, f, protocol=5)

size_kb = out.stat().st_size / 1024
print(f"\nSaved {out.name}  ({size_kb:.0f} KB)  total {time.perf_counter()-t0:.1f}s")
print("Commit demo_cache.pkl to avoid recomputing on cold start.")
