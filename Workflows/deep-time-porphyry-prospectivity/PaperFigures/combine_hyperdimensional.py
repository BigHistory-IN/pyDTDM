# -*- coding: utf-8 -*-
"""Regenerate the hyperdimensional prospectivity map from the CORRECTED models:
spatial prospectivity x max-over-time spatiotemporal prospectivity, regridded to the
spatial grid. Writes cfg.hyperdimensional.combined_nc (used by Figure 1 bottom panel)."""
import os, numpy as np, figlib as F

cfg = F.configure(F.load_config("config.yaml"))
sp, sx, sy = F.open_prosp(cfg["spatial"]["prospectivity_nc"])              # Longitude/Latitude (~0.05 deg, NAM)
st, tx, ty = F.open_prosp(cfg["spatiotemporal"]["max_prospectivity_nc"])  # lon/lat (0.25 deg, global)

# put ST max on the spatial grid, then multiply (both in [0,1])
st = st.rename({tx: sx, ty: sy}).sortby([sx, sy])
st_on_sp = st.interp({sx: sp[sx], sy: sp[sy]})
combined = (sp * st_on_sp).rename("hyperdimensional")

out = cfg["hyperdimensional"]["combined_nc"]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
combined.to_dataset(name="hyperdimensional").to_netcdf(out)
v = combined.values
print(f"WROTE {out}  dims={dict(combined.sizes)}  "
      f"finite={int(np.isfinite(v).sum())}  range=[{np.nanmin(v):.3g}, {np.nanmax(v):.3g}]")
