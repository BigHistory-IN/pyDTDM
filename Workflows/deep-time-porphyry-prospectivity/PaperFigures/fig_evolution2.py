# -*- coding: utf-8 -*-
"""Deep-time evolution — two reconstructed maps (also used as animation frames).
Figure A: crustal thickness (on top of continents) + subducted carbonate overlay
          + plate-motion vectors.
Figure B: subducted sediment + convergence rate with direction (arrows)
          + spatiotemporal prospectivity.
Clean plate topologies ONLY (trenches + teeth + ridges; NO misc_boundaries, which
caused cross-cutting clutter); subduction teeth drawn light so the field shows.
Plate model = alfonso2024 (verified against present-day geography)."""
import os, numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import figlib as F

HERE = os.path.dirname(__file__); OUT = f"{HERE}/out"; os.makedirs(OUT, exist_ok=True)
SRC = "<DATA_ROOT>/Raw/source_data"
EXT = [-184, -78, 0, 82]; PROJ = ccrs.Orthographic(-118, 35)
PCT = ccrs.PlateCarree()
_slices = pd.read_parquet(f"{HERE}/data/grid_slices.parquet")
CARB_OVL = mcolors.LinearSegmentedColormap.from_list("cov", ["#f6c47a", "#e8852f", "#b5341b", "#6e150c"])

def _g(path):
    d = xr.open_dataset(path); v = [c for c in d.data_vars if "spatial_ref" not in c][0]; da = d[v]
    xc = "x" if "x" in da.dims else ("lon" if "lon" in da.dims else list(da.dims)[-1])
    yc = "y" if "y" in da.dims else ("lat" if "lat" in da.dims else list(da.dims)[0])
    return da, da[xc], da[yc]

def continents(ax, gp):
    ax.set_rasterization_zorder(2.5)
    gp.plot_continents(ax, facecolor="#d6d2c9", alpha=0.9, lw=0, zorder=0.5)

def clean_boundaries(ax, gp, teeth="#a6a6a6"):
    """Trenches + teeth (light) + ridges only — NO misc_boundaries/transforms."""
    gp.plot_ridges(ax, color="#c0392b", lw=0.5, alpha=0.65, zorder=2.0)
    gp.plot_trenches(ax, color="#5a5a5a", lw=0.7, alpha=0.9, zorder=2.1)
    gp.plot_subduction_teeth(ax, color=teeth, zorder=2.1)

def velocity(ax, model, time, color="#15324f", step=11):
    gl, ga = np.meshgrid(np.arange(EXT[0]+8, EXT[1]-4, step), np.arange(EXT[2]+6, EXT[3]-4, 10))
    gl, ga = gl.ravel(), ga.ravel()
    v = model.get_point_velocities(gl, ga, time, delta_time=1.0)
    qk = ax.quiver(gl, ga, v[:, 1], v[:, 0], transform=PCT, color=color, scale=950, width=0.0038,
                   alpha=0.85, zorder=5)
    ax.quiverkey(qk, 0.14, 0.06, 5, "5 cm yr$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 9})

def deposits(ax, time, w=2.5):
    dd = F.reconstruct_deposits(time); dd = dd[(dd.age_ma >= time-w) & (dd.age_ma <= time+w)]
    if len(dd):
        F.sized_deposits(ax, dd.rlon.values, dd.rlat.values, dd.tonnage_mt.values, PCT,
                         edge="#ffe08a", halo="#2a2a2a", lw=0.9, zorder=6)
    return len(dd)

def _newmap():
    ax = plt.figure(figsize=(8.5, 8)).add_subplot(111, projection=PROJ)
    ax.set_extent(EXT, crs=PCT); return ax

def make_figA(time):
    """Two clean panels: (left) crustal thickness; (right) subducted carbonate.
    Both with plate-motion vectors, deposits and light topology."""
    F.set_style(); gp = F.gplot_at(time); gp.time = time; model, _ = F.get_model()
    fig = plt.figure(figsize=(14, 7.2))
    axL = fig.add_subplot(1, 2, 1, projection=PROJ); axL.set_extent(EXT, crs=PCT)
    axR = fig.add_subplot(1, 2, 2, projection=PROJ); axR.set_extent(EXT, crs=PCT)
    # LEFT: crustal thickness (on top of continents -> visible)
    continents(axL, gp)
    cr, cx, cy = _g(f"{SRC}/CrustalThickness/crustal_thickness_{time}Ma.nc")
    z = cr.values; z = z/1000.0 if np.nanmax(z) > 200 else z
    im = axL.pcolormesh(cx, cy, z, cmap=F.cmc.lipari, vmin=0, vmax=45, transform=PCT, shading="auto", zorder=1)
    clean_boundaries(axL, gp, teeth="#b0b0b0"); velocity(axL, model, time); deposits(axL, time)
    fig.colorbar(im, ax=axL, shrink=0.5, pad=0.02, location="left", extend="max").set_label("Crustal thickness (km)", fontsize=11)
    axL.set_title("Crustal thickness & plate motion", fontsize=13.5, pad=6)
    axL.text(0.03, 0.97, "a", transform=axL.transAxes, fontsize=16, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=9)
    # RIGHT: subducted carbonate
    continents(axR, gp)
    cb_, bx, by = _g(f"{SRC}/CarbonateThickness/uncompacted_carbonate_thickness_{time}Ma.nc")
    cov = axR.pcolormesh(bx, by, np.where(cb_.values >= 0, cb_.values, np.nan), cmap=F.cmc.lapaz,
                         vmin=0, vmax=300, transform=PCT, shading="auto", zorder=1)
    clean_boundaries(axR, gp, teeth="#b0b0b0"); velocity(axR, model, time); deposits(axR, time)
    fig.colorbar(cov, ax=axR, shrink=0.5, pad=0.02, extend="max").set_label("Subducted carbonate thickness (m)", fontsize=11)
    axR.set_title("Subducted carbonate & plate motion", fontsize=13.5, pad=6)
    axR.text(0.03, 0.97, "b", transform=axR.transAxes, fontsize=16, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=9)
    fig.suptitle(f"Crustal architecture, carbonate flux & plate motion — {time} Ma", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.04, wspace=0.18)
    return fig

def make_figB(time, prosp_df=None):
    """Two clean panels: (left) sediment + convergence rate & direction;
    (right) spatiotemporal prospectivity + plate motion.
    prosp_df: optional DataFrame (lon, lat, 'age (Ma)', 'Prospectivity Score') used
    for the prospectivity panel across animation frames; defaults to grid_slices."""
    src = prosp_df if prosp_df is not None else _slices
    F.set_style(); gp = F.gplot_at(time); gp.time = time; model, _ = F.get_model()
    fig = plt.figure(figsize=(14, 7.2))
    axL = fig.add_subplot(1, 2, 1, projection=PROJ); axL.set_extent(EXT, crs=PCT)
    axR = fig.add_subplot(1, 2, 2, projection=PROJ); axR.set_extent(EXT, crs=PCT)

    # --- LEFT: sediment + convergence rate (colour) + direction (arrows) ---
    continents(axL, gp)
    sd, sx, sy = _g(f"{SRC}/SedimentThickness/sed_thick_0.1d_{time}.nc")
    im = axL.pcolormesh(sx, sy, np.where(sd.values > 1, sd.values, np.nan), cmap="YlGn", vmin=0, vmax=1000,
                        transform=PCT, shading="auto", zorder=1)
    sdz = np.asarray(model.tessellate_subduction_zones(time, np.deg2rad(0.4), ignore_warnings=True,
                     output_convergence_velocity_components=True))
    lon, lat, vel, ang = sdz[:, 0], sdz[:, 1], sdz[:, 2], sdz[:, 3]
    conv = np.abs(vel) * np.cos(np.radians(ang))
    cv = axL.scatter(lon, lat, c=conv, cmap="magma", vmin=0, vmax=12, s=11, marker="o", transform=PCT,
                     zorder=2.5, linewidths=0)
    clean_boundaries(axL, gp, teeth="#b0b0b0"); velocity(axL, model, time, color="#15324f"); deposits(axL, time)
    fig.colorbar(im, ax=axL, shrink=0.5, pad=0.02, location="left", extend="max").set_label("Sediment thickness (m)", fontsize=11)
    fig.colorbar(cv, ax=axL, shrink=0.5, pad=0.02, extend="max").set_label("Convergence rate (cm yr$^{-1}$)", fontsize=11)
    axL.set_title("Slab sediment & convergence", fontsize=13.5, pad=6)
    axL.text(0.03, 0.97, "a", transform=axL.transAxes, fontsize=16, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=9)

    # --- RIGHT: spatiotemporal prospectivity + plate motion ---
    continents(axR, gp)
    sl = src[src["age (Ma)"] == time].dropna(subset=["Prospectivity Score"])
    pm = None
    if len(sl):
        pm = axR.scatter(sl["lon"], sl["lat"], c=sl["Prospectivity Score"], cmap=F.cmc.batlow, vmin=0, vmax=1,
                         s=12, marker="s", transform=PCT, zorder=1.6, linewidths=0)
    clean_boundaries(axR, gp, teeth="#b0b0b0"); velocity(axR, model, time, color="#15324f"); deposits(axR, time)
    if pm is not None:
        fig.colorbar(pm, ax=axR, shrink=0.5, pad=0.02).set_label("Spatiotemporal prospectivity", fontsize=11)
    axR.set_title("Spatiotemporal prospectivity & plate motion", fontsize=13.5, pad=6)
    axR.text(0.03, 0.97, "b", transform=axR.transAxes, fontsize=16, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=9)
    fig.suptitle(f"Slab inputs, convergence & prospectivity — {time} Ma", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.04, wspace=0.18)
    return fig

if __name__ == "__main__":
    import sys
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 55
    fa = make_figA(t); fa.savefig(f"{OUT}/figA_{t}Ma.png", dpi=190, bbox_inches="tight"); fa.savefig(f"{OUT}/figA_{t}Ma.svg")
    fb = make_figB(t); fb.savefig(f"{OUT}/figB_{t}Ma.png", dpi=190, bbox_inches="tight"); fb.savefig(f"{OUT}/figB_{t}Ma.svg")
    print("WROTE figA/figB at", t, "Ma")
