# Paper figures & animations — config-driven workflow

Regenerates the figures for *Plate reorganisation and carbonate subduction create transient
porphyry copper fertility windows* from the prospectivity-workflow outputs, in the project's
plotting style. Plate model = **Alfonso et al. (2024)**.

**Single source of truth: `config.yaml`** — every input path (spatial / spatiotemporal
outputs, the plate model, deep-time fields, deposits), the plotting style, and the output
directory live there. Scripts call `figlib.configure(figlib.load_config("config.yaml"))` and
read `cfg[...]` — no hard-coded paths. `figlib.py` is the single-source plotting +
reconstruction module.

> Set the placeholder paths in `config.yaml` (`<DATA_ROOT>`, `<PATH_TO>`) to your own
> locations before running.

## Environment

gplately, plate_model_manager, cartopy, cmcrameri, xarray, geopandas (see the top-level
`requirements.txt`). On macOS, if PROJ / Cairo errors appear:

```bash
export PROJ_LIB=$CONDA_PREFIX/share/proj
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

## Run order (from this folder)

```bash
python combine_hyperdimensional.py   # spatial × max-spatiotemporal → data/combined_hyperdimensional.nc
python fig1_framework.py             # → out/Figure1.png + .svg
python fig2_models.py                # → out/Figure2.png + .svg
python fig_evolution2.py 55          # reconstructed-parameter frame at 55 Ma
python make_evolution_animation.py 2 # reconstruction animation (step in Myr)
python supp_figs.py                  # supplementary figures
```

## What each figure shows

- **Figure 1** — framework. Present-day spatial-data cascade; reconstructed globes
  (100 / 60 / 20 Ma, carbonate + deposits, Alfonso et al. 2024, closed topologies); and the
  hyperdimensional map (spatial × max-spatiotemporal).
- **Figure 2** — models. (a, b) spatial & max-spatiotemporal prospectivity maps;
  (c, d) top-5 feature importance (spatial 138-feature, spatiotemporal 31-feature);
  (e, f) block-CV success-rate / recall@K curves (spatial-block for the spatial model,
  space–time-block for the spatiotemporal model).

## Notes

- `figlib.py` consolidates: config loader, plotting style, Alfonso et al. (2024)
  reconstruction helpers (`get_model`, `gplot_at`, `reconstruct_deposits`), deposit markers,
  prospectivity-map helpers (`open_prosp`, `plot_prosp_map`), and feature-label prettifiers
  (`nice_spatial`, `clean_st`).
- Success-rate **curves** are read from the out-of-fold block-CV outputs emitted by the
  prospectivity notebooks (defensible, not interpolated).
- License: code — PolyForm Noncommercial 1.0.0; figures / data — CC BY-NC 4.0; commercial use
  via **Geonome Pty Ltd** (see the workflow `NOTICE`).
