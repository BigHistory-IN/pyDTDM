# -*- coding: utf-8 -*-
"""One pass over the spatiotemporal prediction grid to extract:
  (1) full grid slices at selected ages  -> data/grid_slices.parquet  (for param/prospectivity maps)
  (2) trajectories for major deposits     -> data/deposit_trajectories.csv (for time series)
No model is re-run; reads saved predictions only."""
import numpy as np, pandas as pd, os
SRC = ("./"
       "Spatiotemporal/spatiotemporal_grid_predictions_latest.csv")
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

SLICE_AGES = [70, 60, 55, 50, 40, 30, 25, 20, 15, 10, 5, 0]
MAPCOLS = ["lon", "lat", "age (Ma)", "Prospectivity Score", "carbonate_thickness (m)",
           "subducted_carbonates_volume (m)", "crustal_thickness_mean (m)",
           "convergence_rate (cm/yr)", "convergence_rate_parallel (cm/yr)",
           "trench_velocity (cm/yr)", "sediment_thickness (m)"]
DEPOSITS = {  # name: (lon, lat, age)
    "Safford": (-109.628, 32.946, 48), "Morenci-Metcalf": (-109.361, 33.104, 56),
    "Continental/Butte": (-112.511, 46.017, 60), "Bingham": (-112.154, 40.529, 37),
    "Chino": (-108.070, 32.793, 56), "Glacier Peak": (-120.979, 48.198, 21),
    "Bagdad": (-113.212, 34.586, 72), "Ray": (-110.983, 33.164, 69)}
TRAJCOLS = ["present_lon", "present_lat", "age (Ma)", "Prospectivity Score",
            "crustal_thickness_mean (m)", "subducted_carbonates_volume (m)",
            "carbonate_thickness (m)", "convergence_rate (cm/yr)",
            "convergence_rate_parallel (cm/yr)", "convergence_obliquity (degrees)"]
TOL = 0.25

slices, traj = [], []
usecols = sorted(set(MAPCOLS) | set(TRAJCOLS))
for chunk in pd.read_csv(SRC, usecols=usecols, chunksize=1_000_000):
    s = chunk[chunk["age (Ma)"].isin(SLICE_AGES)]
    if len(s):
        slices.append(s[[c for c in MAPCOLS if c in s.columns]].copy())
    for name, (lon, lat, _) in DEPOSITS.items():
        d2 = (chunk["present_lon"]-lon)**2 + (chunk["present_lat"]-lat)**2
        m = chunk[d2 < TOL**2].copy()
        if len(m):
            m["deposit"] = name; m["_d2"] = d2[d2 < TOL**2].values
            traj.append(m[[c for c in TRAJCOLS if c in m.columns] + ["deposit", "_d2"]])

gs = pd.concat(slices); gs.to_parquet(f"{OUT}/grid_slices.parquet");
print("grid_slices:", gs.shape, "ages:", sorted(gs["age (Ma)"].unique()))

tj = pd.concat(traj).sort_values("_d2").drop_duplicates(["deposit", "age (Ma)"]).drop(columns="_d2")
for name, (lon, lat, age) in DEPOSITS.items():
    tj.loc[tj.deposit == name, "mineralization_age"] = age
tj = tj.sort_values(["deposit", "age (Ma)"])
tj.to_csv(f"{OUT}/deposit_trajectories.csv", index=False)
print("trajectories:", tj.shape, "| deposits:", tj.deposit.nunique())
