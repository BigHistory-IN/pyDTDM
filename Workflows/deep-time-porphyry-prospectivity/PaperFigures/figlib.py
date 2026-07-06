# -*- coding: utf-8 -*-
"""Single-source plotting + reconstruction helpers for the Geology paper figures.

Config-driven: call ``configure(load_config("config.yaml"))`` once; every path
(plate model, deep-time fields, deposits, kalpa outputs) then comes from the config —
no hard-coded paths in the figure scripts. Built on the author's own notebook style
(gplately PlotTopologies, cmcrameri colormaps, Orthographic globes, gold deposits).
Plate model = alfonso2024 (loaded from the local model dir; no network fetch)."""
import os, numpy as np, pandas as pd, xarray as xr, geopandas as gpd, yaml
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gplately
from plate_model_manager import PlateModelManager
import cmcrameri.cm as cmc

# ---- config ----------------------------------------------------------------
CFG = None
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def configure(cfg):
    """Set the active config; resets the cached plate model so paths take effect."""
    global CFG, _MODEL
    CFG = cfg
    _MODEL = None
    return cfg

def _cfg():
    if CFG is None:
        raise RuntimeError("figlib not configured — call figlib.configure(load_config('config.yaml')) first")
    return CFG

# ---- shared scientific style -----------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "figure.dpi": 150, "savefig.dpi": 350,
        "savefig.bbox": "tight", "figure.facecolor": "white", "savefig.facecolor": "white",
    })

# colormaps per parameter (author's choices)
CMAPS = {
    "Prospectivity Score": (cmc.batlow, 0, 1, "Spatiotemporal prospectivity"),
    "carbonate_thickness (m)": (cmc.lapaz, 0, 300, "Carbonate thickness (m)"),
    "crustal_thickness_mean (m)": (cmc.lipari, 30000, 55000, "Crustal thickness (m)"),
    "convergence_rate (cm/yr)": ("magma", 0, 15, "Convergence rate (cm yr$^{-1}$)"),
    "sediment_thickness (m)": ("Wistia", 0, 3000, "Sediment thickness (m)"),
}

# ---- plate model + reconstruction (accuracy-critical) ----------------------
_MODEL = None
def get_model():
    """alfonso2024 from the local model dir (cfg.plate_model). Cached."""
    global _MODEL
    if _MODEL is None:
        pm_cfg = _cfg()["plate_model"]
        pm = PlateModelManager().get_model(pm_cfg["name"], data_dir=pm_cfg["data_dir"])
        model = gplately.PlateReconstruction(pm.get_rotation_model(), pm.get_topologies(),
                                             pm.get_static_polygons())
        _MODEL = (model, pm)
    return _MODEL

def gplot_at(time):
    model, pm = get_model()
    return gplately.PlotTopologies(model, coastlines=pm.get_coastlines(),
                                   continents=pm.get_continental_polygons(), time=time)

def deposits_df():
    g = gpd.read_file(_cfg()["deposits"]["file"])
    return g[["name", "latitude", "longitude", "age_ma", "tonnage_mt"]].copy()

def reconstruct_deposits(time):
    """Reconstruct present-day deposits to `time` Ma. gplately Points.reconstruct
    RETURNS reconstructed coords (functional; does not mutate the object)."""
    d = deposits_df()
    d = d[d["tonnage_mt"] > 0].dropna(subset=["longitude", "latitude"])
    model, _ = get_model()
    pts = gplately.Points(model, d["longitude"].values, d["latitude"].values)
    rlons, rlats = pts.reconstruct(time, return_array=True)
    return d.assign(rlon=rlons, rlat=rlats)

# ---- deposit markers --------------------------------------------------------
def deposit_sizes(ton):
    t = np.where(np.asarray(ton, float) > 0, np.asarray(ton, float), 0.3)
    return 6 + 2.6 * np.sqrt(t)

def sized_deposits(ax, lons, lats, ton, transform, edge="white", halo="#141414", lw=1.0, zorder=6):
    sz = deposit_sizes(ton)
    ax.scatter(lons, lats, s=sz * 1.4, facecolor="none", edgecolor=halo,
               linewidth=lw + 0.9, alpha=0.6, transform=transform, zorder=zorder - 0.5)
    ax.scatter(lons, lats, s=sz, facecolor="none", edgecolor=edge, linewidth=lw,
               alpha=0.95, transform=transform, zorder=zorder)

def deposit_size_legend(ax, vals=(1, 10, 100, 1000), edge="white", loc="lower left",
                        title="Contained Cu (Mt)", fontsize=8.5):
    import matplotlib.lines as ml
    h = [ml.Line2D([], [], marker="o", ls="", mfc="none", mec=edge, mew=1.0,
                   ms=np.sqrt(deposit_sizes(v))) for v in vals]
    lg = ax.legend(h, [str(v) for v in vals], loc=loc, title=title, fontsize=fontsize,
                   title_fontsize=fontsize + 0.5, labelspacing=1.25, framealpha=0.88,
                   borderpad=0.8, handletextpad=1.0)
    lg.set_zorder(11); return lg

def base_topologies(ax, gp, lw=0.12, alpha=0.6):
    ax.set_rasterization_zorder(4)
    gp.plot_continents(ax, facecolor="#e6e3dd", alpha=0.9, lw=0, zorder=1)
    gp.plot_coastlines(ax, color="#9a948b", alpha=0.7, lw=lw, zorder=1.5)
    gp.plot_trenches(ax, color="k", lw=0.9, alpha=alpha, zorder=2.5)
    gp.plot_subduction_teeth(ax, color="k", alpha=alpha, zorder=2.6)
    gp.plot_ridges(ax, color="#b8412e", lw=0.5, alpha=0.6, zorder=2.4)

# ---- prospectivity NetCDF helpers (absorb coord-name / extent differences) --
def open_prosp(path):
    """Open a prospectivity NetCDF -> (DataArray, xname, yname). Handles both
    Longitude/Latitude (spatial) and lowercase lon/lat (spatiotemporal max)."""
    ds = xr.open_dataset(path)
    da = ds[[v for v in ds.data_vars if "spatial_ref" not in v][0]]
    xn = next((c for c in ("Longitude", "lon", "longitude", "lon_t") if c in da.coords), da.dims[-1])
    yn = next((c for c in ("Latitude", "lat", "latitude", "lat_t") if c in da.coords), da.dims[0])
    return da, xn, yn

def plot_prosp_map(ax, da, xn, yn, cmap=None, vmin=0, vmax=1, extent=None, transform=None):
    cmap = cmap if cmap is not None else cmc.batlow
    tr = transform if transform is not None else ccrs.PlateCarree()
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax.pcolormesh(da[xn], da[yn], da.values, cmap=cmap, vmin=vmin, vmax=vmax,
                         transform=tr, shading="auto", zorder=1)

# ---- feature-importance label prettifiers ----------------------------------
_LAYER = {"GeophysicsGravity_Up30km_HGM": "Gravity 30 km-up (HGM)", "GeophysicsGravity_Up30km": "Gravity (30 km up)",
          "GeophysicsGravity_HGM": "Gravity (HGM)", "GeophysicsGravity": "Bouguer gravity",
          "GeophysicsMag_RTP_VD": "Magnetic RTP–VD", "GeophysicsMag_RTP_HGM": "Magnetic RTP–HGM",
          "GeophysicsMag_RTP_USCanada": "Magnetic RTP", "GeophysicsMag": "Magnetic anomaly",
          "sio2_silica_index": "Silica index", "ferric_iron": "Ferric iron", "ferrous_iron": "Ferrous iron",
          "hydrothermal_alteration": "Hydrothermal alt.", "al2o3_alteration": "Al₂O₃ alteration",
          "al2o3_laterite_argilic": "Al₂O₃ laterite"}
_DERIV = {"x_grad": "E–W gradient", "y_grad": "N–S gradient", "magnitude_grad": "gradient mag."}
_STAT = {"std": "variability", "median": "median", "max": "max", "min": "min", "mean": "mean",
         "correlation": "texture", "contrast": "texture"}
def nice_spatial(f):
    if "nearest_distance" in f:
        nm = (f.replace("_nearest_distance", "").replace("Litho_", "").replace("Faults_", "Faults ")
               .replace("Geologic_contacts", "geologic contacts").replace("_", " "))
        return f"Distance to {nm.strip().lower()}"
    base, rest = None, f
    for k in sorted(_LAYER, key=len, reverse=True):
        if f.startswith(k):
            base = _LAYER[k]; rest = f[len(k):].strip("_"); break
    if base is None:
        return f.replace("_", " ")
    d = next((v for k, v in _DERIV.items() if k in rest), "")
    s = next((v for k, v in _STAT.items() if rest.endswith(k)), "")
    return base + (f", {d}" if d else "") + (f" ({s})" if s else "")

def clean_st(name):
    """Strip unit suffixes from spatiotemporal feature names for tidy labels."""
    out = name
    for u in (" (m)", " (cm/yr)", " (m^2/yr)", " (km/Myr)", " (degrees)", " (Ma)", " (km)"):
        out = out.replace(u, "")
    return out.replace("_", " ")
