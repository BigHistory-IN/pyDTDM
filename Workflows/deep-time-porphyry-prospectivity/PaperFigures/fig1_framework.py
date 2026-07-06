# -*- coding: utf-8 -*-
"""Figure 1 — Spatiotemporal prospectivity framework (config-driven).
Left  : cascade of present-day SPATIAL DATA layers.
Right : three reconstructed globes (CLOSED plate topologies, alfonso2024) through time.
Flow  : Feature Engineering / Deep Time Data Mining -> PU ML Framework ->
        Spatial / Spatiotemporal -> Hyperdimensional Prospectivity.
Bottom: hyperdimensional prospectivity map = spatial x max-spatiotemporal (CORRECTED models).
All paths/style from config.yaml."""
import os, numpy as np, xarray as xr
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import cartopy.crs as ccrs, cartopy.feature as cfeature
import figlib as F

cfg = F.configure(F.load_config("config.yaml")); F.set_style()
S = cfg["style"]; EXT = S["nam_extent"]; INK = S["colors"]["ink"]
OUT = cfg["out_dir"]; os.makedirs(OUT, exist_ok=True)
PAN = f"{OUT}/fig1_panels"; os.makedirs(PAN, exist_ok=True)
CARB = cfg["deeptime"]["carbonate_dir"]
BLUE, PURPLE, ORANGE = "#bcd3ea", "#d9cfe9", "#fbe1c6"
CARB_CMAP = LinearSegmentedColormap.from_list(
    "carb", ["#0b1f3a", "#15355f", "#2e6f97", "#7fb0a3", "#e8b455", "#d9612b", "#9e2a18"])

fig = plt.figure(figsize=(12.5, 13.6))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 100); bg.axis("off")

def box(x, y, w, h, text, fc, fs=15, ec="#4a4a4a", weight="bold", tc=INK):
    bg.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.4, zorder=3))
    bg.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight, zorder=4, linespacing=1.1)

def arrow(x0, y0, x1, y1, rad=0.0, lw=2.4, color="#3a3a3a"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2))

# ============================ LEFT: present-day spatial data ================
bg.add_patch(FancyBboxPatch((3, 70.5), 44, 27.5, boxstyle="round,pad=0.3", fc="white", ec="#4a4a4a", lw=1.6, zorder=1))
bg.text(2.0, 84, "Present-day Spatial Data", rotation=90, va="center", ha="center", fontsize=16, fontweight="bold", color=INK)
LAYERS = cfg["spatial_rasters"]   # list of {file, cmap, pct, title}

def draw_layer(ax, layer, cbar_ax=None, title_box=True):
    ax.set_rasterization_zorder(2)
    d = xr.open_dataset(layer["file"])
    v = [c for c in d.data_vars if "spatial_ref" not in c][0]; da = d[v]
    stride = max(1, max(da.sizes["x"], da.sizes["y"]) // 500)
    da = da.isel(x=slice(None, None, stride), y=slice(None, None, stride))
    arr = da.values.astype("float32"); pct = layer["pct"]
    vmin, vmax = np.nanpercentile(arr, pct[0]), np.nanpercentile(arr, pct[1])
    ax.set_extent(EXT, crs=ccrs.PlateCarree()); ax.add_feature(cfeature.LAND, facecolor="#efeee9", zorder=0)
    im = ax.pcolormesh(da["x"], da["y"], arr, cmap=layer["cmap"], vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="#777", zorder=1.5)
    for s in ax.spines.values(): s.set_edgecolor("#3a3a3a"); s.set_linewidth(1.1)
    if title_box:
        ax.text(0.5, 1.02, layer["title"], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK, zorder=12,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#9a9a9a", lw=0.6, alpha=0.97))
    if cbar_ax is not None:
        cb = fig.colorbar(im, cax=cbar_ax); cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_linewidth(0.5); cb.set_ticks([vmin, vmax]); cb.set_ticklabels(["lo", "hi"])
    return im

CASC = [(0.30, 0.852, 0.135, 0.085), (0.235, 0.808, 0.150, 0.095),
        (0.165, 0.762, 0.165, 0.105), (0.085, 0.714, 0.185, 0.118)]
for z, (layer, (x, y, w, h)) in enumerate(zip(LAYERS, CASC)):
    ax = fig.add_axes([x, y, w, h], projection=ccrs.PlateCarree()); ax.set_zorder(2 + z)
    cax = fig.add_axes([x + w + 0.004, y + 0.012, 0.008, h - 0.024]); cax.set_zorder(2 + z)
    draw_layer(ax, layer, cbar_ax=cax)

# ============================ RIGHT: deep-time globes ========================
bg.add_patch(FancyBboxPatch((53, 70.5), 44, 27.5, boxstyle="round,pad=0.3", fc="white", ec="#4a4a4a", lw=1.6, zorder=1))
bg.text(98.0, 84, "Deep Time Data", rotation=90, va="center", ha="center", fontsize=16, fontweight="bold", color=INK)
bg.annotate("", xy=(0.905, 0.965), xytext=(0.575, 0.915), xycoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#3a3a3a"))
bg.text(0.70, 0.952, "Time", transform=fig.transFigure, fontsize=15, fontstyle="italic", fontweight="bold", color=INK, ha="center")
GLOBE_TIMES = S["globe_times"]; TOL = cfg["deposits"]["age_tolerance_ma"]
GPOS = [(0.565, 0.752, 0.155, 0.135), (0.66, 0.782, 0.155, 0.135), (0.755, 0.812, 0.155, 0.135)]
def draw_globe(ax, t):
    ax.set_rasterization_zorder(2); ax.set_global()
    gp = F.gplot_at(t); gp.time = t
    nc = xr.open_dataarray(f"{CARB}/uncompacted_carbonate_thickness_{t}Ma.nc")
    ax.pcolormesh(nc["lon"], nc["lat"], np.where(nc.values >= 0, nc.values, np.nan),
                  cmap=CARB_CMAP, vmin=0, vmax=S["carb_vmax"], transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    gp.plot_continents(ax, facecolor="#cfcbc3", alpha=0.95, lw=0, zorder=1.2)
    gp.plot_coastlines(ax, color="#8a8580", lw=0.2, alpha=0.7, zorder=1.3)
    gp.plot_trenches(ax, color="k", lw=0.5, zorder=1.5)
    gp.plot_subduction_teeth(ax, color="k", zorder=1.5)
    gp.plot_ridges(ax, color="k", lw=0.3, alpha=0.7, zorder=1.5)
    dd = F.reconstruct_deposits(t); dd = dd[dd["age_ma"] >= t - TOL]
    ax.scatter(dd["rlon"], dd["rlat"], s=np.sqrt(dd["tonnage_mt"]) * 0.9, facecolor="#ffd966",
               edgecolor="#2a2a2a", linewidth=0.4, alpha=0.9, transform=ccrs.PlateCarree(), zorder=3)
    ax.set_title(f"{t} Ma", fontsize=12.5, fontweight="bold", pad=1)
for t, (x, y, w, h) in zip(GLOBE_TIMES, GPOS):
    draw_globe(fig.add_axes([x, y, w, h], projection=ccrs.Orthographic(-90, 20)), t)
cax = fig.add_axes([0.62, 0.722, 0.26, 0.0105])
cbg = fig.colorbar(mpl.cm.ScalarMappable(cmap=CARB_CMAP, norm=mpl.colors.Normalize(0, S["carb_vmax"])),
                   cax=cax, orientation="horizontal", extend="max")
cbg.set_label("Carbonate thickness (m)", fontsize=10); cbg.ax.tick_params(labelsize=8.5)

# ============================ FLOW ==========================================
box(25, 65, 26, 5.2, "Feature Engineering", BLUE); box(75, 65, 26, 5.2, "Deep Time Data Mining", PURPLE)
arrow(25, 70.4, 25, 67.7); arrow(75, 70.4, 75, 67.7)
box(50, 57.5, 70, 5.6, "Positive Unlabelled Machine Learning Framework", ORANGE, fs=16)
arrow(25, 62.3, 30, 60.4); arrow(75, 62.3, 70, 60.4)
box(25, 50, 27, 5.2, "Spatial Prospectivity", ORANGE); box(75, 50, 30, 5.2, "Spatiotemporal Prospectivity", ORANGE)
arrow(38, 54.6, 30, 52.7); arrow(62, 54.6, 70, 52.7)
box(50, 42.5, 30, 6.2, "Hyperdimensional\nProspectivity", ORANGE)
arrow(28, 47.5, 44, 44.2, rad=-0.12); arrow(72, 47.5, 56, 44.2, rad=0.12)

# ============================ BOTTOM: hyperdimensional map ===================
arrow(50, 39.0, 50, 36.8)
axm = fig.add_axes([0.235, 0.03, 0.55, 0.335], projection=ccrs.PlateCarree()); axm.set_rasterization_zorder(2)
hd = xr.open_dataset(cfg["hyperdimensional"]["combined_nc"]); cm_ = hd[list(hd.data_vars)[0]]
perc = cfg["hyperdimensional"]["percentiles"]
thr = sorted(float(cm_.quantile(p/100).values) for p in perc)
names = ['50–100%', '20–50%', '10–20%', '5–10%', '2–5%', '1–2%', 'Top 1%']
colors = ["#f5f6d6", '#ffeda0', "#efa964", "#c97240", "#B24920", "#622316", "#000000"]
cmap_d = ListedColormap(colors); norm = BoundaryNorm(thr, cmap_d.N)
axm.set_extent(EXT, crs=ccrs.PlateCarree()); axm.add_feature(cfeature.LAND, facecolor="#f3f1ec", zorder=0)
im = axm.pcolormesh(cm_["Longitude"], cm_["Latitude"], cm_.values, cmap=cmap_d, norm=norm,
                    transform=ccrs.PlateCarree(), shading="auto", zorder=1)
axm.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#777", zorder=3)
axm.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#9a9a9a", zorder=3)
dep = F.deposits_df().dropna(subset=["longitude", "latitude"]); dep = dep[dep.tonnage_mt > 0]
F.sized_deposits(axm, dep.longitude.values, dep.latitude.values, dep.tonnage_mt.values,
                 ccrs.PlateCarree(), edge="#0d3b66", halo="white", lw=1.0)
gl = axm.gridlines(draw_labels=True, ls=":", lw=0.4, color="#c3c7cb", alpha=0.6)
gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 10}
F.deposit_size_legend(axm, edge="#0d3b66", loc="upper right")
axm.set_title("Hyperdimensional Prospectivity", fontsize=15, fontweight="bold", pad=6)
cb = plt.colorbar(im, ax=axm, fraction=0.03, pad=0.02); cb.set_label("Prospectivity percentile", fontsize=11.5)
cb.set_ticks([(thr[i]+thr[i+1])/2 for i in range(len(thr)-1)]); cb.set_ticklabels(names, fontsize=9.5)

fig.savefig(f"{OUT}/Figure1.png", dpi=S["save_dpi"]); fig.savefig(f"{OUT}/Figure1.svg")
print("WROTE", f"{OUT}/Figure1.png")
