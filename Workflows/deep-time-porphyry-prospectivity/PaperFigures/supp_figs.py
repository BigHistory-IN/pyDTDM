# -*- coding: utf-8 -*-
"""Supplementary figures (rebuilt, visible).
SF1: key present-day spatial predictor layers (rasters), large panels.
SF4: early-discovery efficiency, spatial vs spatiotemporal, full 1-50% range."""
import os, numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs, cartopy.feature as cfeature
import figlib as F

HERE = os.path.dirname(__file__); OUT = f"{HERE}/out"; os.makedirs(OUT, exist_ok=True)
RAW = "<DATA_ROOT>/Raw/Rasters"
ST = "."
F.set_style()
EXT = [-145, -101, 27, 72]; proj = ccrs.PlateCarree()

# ---------------- SF1 : predictor layers ------------------------------------
LAYERS = [
 ("Gravity/GeophysicsGravity.nc", "Bouguer gravity anomaly", "magma", None),
 ("Gravity/GeophysicsGravity_HGM.nc", "Gravity horizontal-gradient magnitude", "cividis", (1, 99)),
 ("Magnetic/GeophysicsMag_RTP_VD.nc", "Magnetic anomaly (RTP, vertical derivative)", "RdBu_r", (2, 98)),
 ("Magnetic/GeophysicsMag_RTP_HGM.nc", "Magnetic anomaly (RTP, HGM)", "cividis", (1, 99)),
 ("MineralIndex/hydrothermal_alteration.nc", "Hydrothermal-alteration index", "YlOrRd", (2, 98)),
 ("MineralIndex/ferric_iron.nc", "Ferric-iron index", "YlOrRd", (2, 98)),
 ("MineralIndex/sio2_silica_index.nc", "SiO$_2$ silica index", "viridis", (2, 98)),
 ("MineralIndex/al2o3_alteration.nc", "Al$_2$O$_3$ alteration index", "YlOrRd", (2, 98)),
]
fig, axes = plt.subplots(4, 2, figsize=(13, 16), subplot_kw={"projection": proj})
axes = axes.ravel()
for ax, (fn, title, cmap, pct) in zip(axes, LAYERS):
    d = xr.open_dataset(f"{RAW}/{fn}")
    v = [x for x in d.data_vars if "spatial_ref" not in x][0]
    da = d[v]
    stride = max(1, max(da.sizes["x"], da.sizes["y"]) // 900)
    da = da.isel(x=slice(None, None, stride), y=slice(None, None, stride))
    arr = da.values.astype("float32")
    if pct:
        vmin, vmax = np.nanpercentile(arr, pct[0]), np.nanpercentile(arr, pct[1])
    else:
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    ax.set_extent(EXT, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#eeece7", zorder=0)
    im = ax.pcolormesh(da["x"], da["y"], arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=proj, shading="auto", rasterized=True, zorder=1)
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#555", zorder=3)
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#888", zorder=3)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=13, pad=5)
    gl = ax.gridlines(draw_labels=True, ls=":", lw=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
fig.suptitle("Supplementary Figure S1 · Present-day spatial predictor layers (Stream A)",
             fontsize=16, fontweight="bold", y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(f"{OUT}/SFigure1.png", dpi=200); fig.savefig(f"{OUT}/SFigure1.svg")
plt.close(fig); print("WROTE SFigure1")

# ---------------- SF4 : early-discovery efficiency (full range) -------------
sp = pd.read_csv(f"{ST}/SpatialProspectivity/success_rate_curve_data_PU_aware.csv")
st = pd.read_csv(f"{ST}/Spatiotemporal/spatiotemporal_success_rate_curve_data.csv")
def derive(sr):
    m = sr.copy(); m["recall"] = m["Success Rate %"]; m["enr"] = m["Success Rate %"]/m["Area %"]
    return m
spm, stm = derive(sp), derive(st)
C_SP, C_ST = "#e07b39", "#1b6ca8"
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.13, wspace=0.22)
a = ax[0]
a.plot(spm["Area %"], spm["enr"], "-o", color=C_SP, lw=2.4, ms=5, label="Spatial only")
a.plot(stm["Area %"], stm["enr"], "-o", color=C_ST, lw=2.6, ms=5.5, label="Spatiotemporal")
a.axhline(1, color="#9aa0a6", ls="--", lw=1.3, label="Random (=1)")
a.set(xlim=(0, 100), xlabel="Cumulative area explored (%)", ylabel="Enrichment (lift over random)")
a.set_title("Enrichment vs area explored", fontsize=14); a.legend(); a.grid(alpha=0.25)
a.text(-0.1, 1.04, "a", transform=a.transAxes, fontsize=16, fontweight="bold")
b = ax[1]
ks = [1, 2, 5, 10, 20, 30, 50]; x = np.arange(len(ks)); w = 0.4
spi, sti = spm.set_index("Area %"), stm.set_index("Area %")
b.bar(x-w/2, [spi.loc[k, "Deposits Found"] for k in ks], w, color=C_SP, ec="white", label="Spatial (of 119)")
b.bar(x+w/2, [sti.loc[k, "Deposits Found"] for k in ks], w, color=C_ST, ec="white", label="Spatiotemporal (of 104)")
for xi, k in zip(x-w/2, ks):
    b.text(xi, spi.loc[k, "Deposits Found"]+1, f"{int(spi.loc[k,'Deposits Found'])}\n{spi.loc[k,'Success Rate %']:.0f}%", ha="center", fontsize=8)
for xi, k in zip(x+w/2, ks):
    b.text(xi, sti.loc[k, "Deposits Found"]+1, f"{int(sti.loc[k,'Deposits Found'])}\n{sti.loc[k,'Success Rate %']:.0f}%", ha="center", fontsize=8, color=C_ST)
b.set(xticks=x, ylim=(0, 135), xlabel="Top-ranked area (%)", ylabel="Deposits captured")
b.set_xticklabels([f"{k}%" for k in ks]); b.set_title("Deposits captured (full range)", fontsize=14)
b.legend(loc="upper left"); b.grid(axis="y", alpha=0.25)
b.text(-0.1, 1.04, "b", transform=b.transAxes, fontsize=16, fontweight="bold")
fig.suptitle("Supplementary Figure S4 · Early-discovery efficiency: spatial vs spatiotemporal",
             fontsize=15, fontweight="bold", y=0.98)
fig.savefig(f"{OUT}/SFigure4.png", dpi=200); fig.savefig(f"{OUT}/SFigure4.svg")
plt.close(fig); print("WROTE SFigure4")
