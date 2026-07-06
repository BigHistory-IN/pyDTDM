import pandas as pd, numpy as np, os
SRC="./Spatiotemporal/spatiotemporal_grid_predictions_latest.csv"
OUT=os.path.join(os.path.dirname(__file__),"data"); os.makedirs(OUT,exist_ok=True)
ages=list(range(0,171,2))
cols=["lon","lat","age (Ma)","Prospectivity Score","convergence_rate (cm/yr)"]
acc=[]
for ch in pd.read_csv(SRC,usecols=cols,chunksize=1_000_000):
    s=ch[ch["age (Ma)"].isin(ages)]
    if len(s): acc.append(s)
d=pd.concat(acc); d.to_parquet(f"{OUT}/anim_grid.parquet")
print("anim_grid:",d.shape,"ages:",sorted(d['age (Ma)'].unique()))
