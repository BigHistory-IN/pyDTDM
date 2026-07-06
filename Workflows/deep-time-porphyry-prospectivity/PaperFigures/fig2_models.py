# -*- coding: utf-8 -*-
"""Figure 2 — Spatial vs spatiotemporal models (CORRECTED: spatial=138 feat,
spatiotemporal=31 feat; HONEST block-CV performance).
(a,b) prospectivity maps · (c,d) top-5 feature importance · (e,f) honest success-rate / recall@K.
All inputs come from config.yaml (the kalpa workflow outputs)."""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import figlib as F

cfg = F.configure(F.load_config("config.yaml")); F.set_style()
S = cfg["style"]; OUT = cfg["out_dir"]; os.makedirs(OUT, exist_ok=True)
EXT = S["nam_extent"]; proj = ccrs.PlateCarree(); CMAP = F.cmc.batlow
C_SP, C_ST, C_RND = S["colors"]["spatial"], S["colors"]["spatiotemporal"], S["colors"]["random"]
NTOP = S["n_top"]
dep = F.deposits_df().dropna(subset=["longitude", "latitude"]); dep = dep[dep.tonnage_mt > 0]

def cv_recall(path, key, k="recall_at_10"):
    m = json.load(open(path)); return 100.0 * m[key][k], 100.0 * m[key]["auc_roc"]
sp_r10, sp_auc = cv_recall(cfg["spatial"]["cv_metrics_json"], cfg["spatial"]["cv_key"])
st_r10, st_auc = cv_recall(cfg["spatiotemporal"]["cv_metrics_json"], cfg["spatiotemporal"]["cv_key"])

fig = plt.figure(figsize=(14, 13.5))
outer = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.72, 0.82], hspace=0.34,
                          left=0.04, right=0.97, top=0.96, bottom=0.055)
gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, outer[0], wspace=0.16)
gs_fi   = gridspec.GridSpecFromSubplotSpec(1, 2, outer[1], wspace=0.62)
gs_perf = gridspec.GridSpecFromSubplotSpec(1, 2, outer[2], wspace=0.22)

def basemap(ax, title, tag):
    ax.set_extent(EXT, crs=proj); ax.add_feature(cfeature.LAND, facecolor="#f3f1ec", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.5, edgecolor="#6f6f6f", zorder=4)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="#9a9a9a", zorder=4)
    ax.add_feature(cfeature.STATES, lw=0.25, edgecolor="#cfcfcf", alpha=0.7, zorder=4)
    F.sized_deposits(ax, dep.longitude.values, dep.latitude.values, dep.tonnage_mt.values, proj, edge="white", lw=0.9)
    gl = ax.gridlines(draw_labels=True, ls=":", lw=0.4, color="#c3c7cb", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {"size": 10}
    ax.set_title(title, fontsize=15.5, pad=7)
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top",
            zorder=9, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

# (a) spatial map, (b) max-spatiotemporal map  (open_prosp absorbs coord-name + global-extent diffs)
axa = fig.add_subplot(gs_maps[0], projection=proj); axa.set_rasterization_zorder(2)
da, xn, yn = F.open_prosp(cfg["spatial"]["prospectivity_nc"])
im = F.plot_prosp_map(axa, da, xn, yn, cmap=CMAP, vmin=0, vmax=1, extent=EXT)
basemap(axa, "Spatial prospectivity", "a")
plt.colorbar(im, ax=axa, shrink=0.62, pad=0.02).set_label("Prospectivity score", fontsize=12)

axb = fig.add_subplot(gs_maps[1], projection=proj); axb.set_rasterization_zorder(2)
da2, xn2, yn2 = F.open_prosp(cfg["spatiotemporal"]["max_prospectivity_nc"])
im2 = F.plot_prosp_map(axb, da2, xn2, yn2, cmap=CMAP, vmin=0, vmax=1, extent=EXT)
basemap(axb, "Maximum spatiotemporal prospectivity", "b")
plt.colorbar(im2, ax=axb, shrink=0.62, pad=0.02).set_label("Max prospectivity score", fontsize=12)
F.deposit_size_legend(axb, edge="#1d1d1f", loc="lower left")   # dark markers: visible on the light legend box

# (c,d) top-5 feature importance (corrected models)
def two_line(lbl):
    if "," in lbl:
        a, b = lbl.split(",", 1); return a.strip() + ",\n" + b.strip()
    w = lbl.split()
    return lbl if len(w) <= 2 else " ".join(w[:(len(w)+1)//2]) + "\n" + " ".join(w[(len(w)+1)//2:])

def fibar(ax, df, labels, color, title, tag):
    ax.barh(range(len(df)), df["importance_mean"], color=color, height=0.62, edgecolor="white", lw=0.6)
    ax.set_yticks(range(len(df))); ax.set_yticklabels([two_line(l) for l in labels], fontsize=10)
    for i, v in enumerate(df["importance_mean"]):
        ax.text(v + ax.get_xlim()[1]*0.012, i, f"{v:.2f}", va="center", fontsize=9, color="#444")
    ax.set_xlabel("Mean feature importance", fontsize=11.5); ax.set_title(title, fontsize=14, pad=6)
    ax.grid(axis="x", alpha=0.25); ax.margins(x=0.13)
    ax.text(-0.50, 1.07, tag, transform=ax.transAxes, fontsize=18, fontweight="bold")

axc = fig.add_subplot(gs_fi[0])
sfi = pd.read_csv(cfg["spatial"]["feature_importance_csv"]).sort_values("importance_mean", ascending=False).head(NTOP).iloc[::-1]
fibar(axc, sfi, [F.nice_spatial(f) for f in sfi["feature"]], C_SP, "Spatial model — top predictors", "c")
axd = fig.add_subplot(gs_fi[1])
tfi = pd.read_csv(cfg["spatiotemporal"]["feature_importance_csv"]).sort_values("importance_mean", ascending=False).head(NTOP).iloc[::-1]
fibar(axd, tfi, [F.clean_st(f) for f in tfi["feature"]], C_ST, "Spatiotemporal model — top predictors", "d")

# (e,f) HONEST success-rate / recall@K curves (block-CV out-of-fold)
spc = pd.read_csv(cfg["spatial"]["success_curve_csv"])              # area_pct, recall_pct
stc = pd.read_csv(cfg["spatiotemporal"]["success_curve_csv"])
axe = fig.add_subplot(gs_perf[0])
axe.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=C_RND, lw=1.3, label="Random")
axe.plot(spc["area_pct"], spc["recall_pct"], color=C_SP, lw=2.5, label="Spatial (spatial-block CV)")
axe.plot(stc["area_pct"], stc["recall_pct"], color=C_ST, lw=2.7, label="Spatiotemporal (space–time-block CV)")
axe.scatter([10, 10], [sp_r10, st_r10], s=46, c=[C_SP, C_ST], ec="white", zorder=6)
axe.annotate(f"{sp_r10:.0f}%", (10, sp_r10), (16, sp_r10-7), color=C_SP, fontweight="bold", fontsize=11)
axe.annotate(f"{st_r10:.0f}%", (10, st_r10), (16, st_r10+3), color=C_ST, fontweight="bold", fontsize=11)
axe.set(xlim=(0, 100), ylim=(0, 102), xlabel="Cumulative area explored (%)", ylabel="Deposits recovered (%)")
axe.set_title("Success-rate curve (honest CV)", fontsize=14, pad=6); axe.legend(loc="lower right", fontsize=9); axe.grid(alpha=0.25)
axe.text(-0.17, 1.05, "e", transform=axe.transAxes, fontsize=18, fontweight="bold")

axf = fig.add_subplot(gs_perf[1])
for c, col, lab in [(spc, C_SP, "Spatial"), (stc, C_ST, "Spatiotemporal")]:
    z = c[c["area_pct"] <= 30]
    axf.plot(z["area_pct"], z["recall_pct"], color=col, lw=2.6, label=lab)
axf.set(xlim=(0, 30), ylim=(0, 102), xlabel="Top-ranked area (%)", ylabel="Recall@K — deposits recovered (%)")
axf.set_title("Recall@K (top 30%)", fontsize=14, pad=6); axf.legend(loc="lower right", fontsize=10); axf.grid(alpha=0.25)
axf.text(-0.17, 1.05, "f", transform=axf.transAxes, fontsize=18, fontweight="bold")

fig.text(0.5, 0.012, f"Honest block-CV: spatial AUC {sp_auc:.0f}%, recall@10% {sp_r10:.0f}%  ·  "
         f"spatiotemporal AUC {st_auc:.0f}%, recall@10% {st_r10:.0f}%", ha="center", fontsize=9, color="#555")
fig.savefig(f"{OUT}/Figure2.png", dpi=S["save_dpi"]); fig.savefig(f"{OUT}/Figure2.svg")
print("WROTE", f"{OUT}/Figure2.png  (spatial top:", list(sfi['feature'].iloc[::-1][:3]), ")")
