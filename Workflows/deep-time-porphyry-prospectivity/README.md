# Deep-Time Porphyry Copper Prospectivity

Companion code for:

> Norbisrath, J.H., Singh, S.P., Singh, U., & Müller, R.D. *Plate reorganisation and
> carbonate subduction create transient porphyry copper fertility windows.* (submitted to
> *Geology*). DOI to be added on publication.

Machine-learning prospectivity mapping for porphyry-copper deposits across North America,
in two complementary framings:

- **Spatial** — present-day feature stack, positive–unlabelled (PU) bagging.
- **Spatiotemporal** — deep-time reconstructed features (plate kinematics, subducting
  carbonate, crustal thickness, …) through the Cenozoic, resolving *transient fertility
  windows* rather than a single static map.

## Contents

| Path | Purpose |
|---|---|
| `PreprocessingPorphyrycoppersample.ipynb`, `preprocessing.ipynb`, `GenerateNetCDF.ipynb` | Assemble and grid the training / prediction feature stacks. |
| `MachineLearningModel.ipynb` | PU-bagging model training utilities. |
| `SpatialProspectivity/1_SpatialProspectivity_Workflow.ipynb` | End-to-end spatial prospectivity (sampling → feature selection → PU model → map). |
| `Spatiotemporal/2_SpatiotemporalProspectivity.ipynb` | Deep-time spatiotemporal prospectivity model. |
| `Spatiotemporal/3_Plotting_Spatiotemporal_Data.ipynb` | Reconstructed parameter & prospectivity maps; animations. |
| `Spatiotemporal/4_Hyperdimensional_Prospectivity.ipynb` | Combined spatial × spatiotemporal ("hyperdimensional") prospectivity. |
| `Spatiotemporal/PlottingDeepTimeData.ipynb` | Deposit time-series and deep-time data figures. |
| `PaperFigures/` | Config-driven scripts (`config.yaml` + `figlib.py`) that regenerate the paper figures. |

## Data & plate model — not included here

This repository is **code only**. The inputs are distributed separately:

- **Analysis-ready data** (training tables, feature grids, result grids) — Zenodo:
  `10.5281/zenodo.XXXXXXX` *(DOI to be added on publication)*.
- **Plate reconstruction model** — **Alfonso et al. (2024)**. All deep-time plate
  kinematics and reconstructed deep-time fields used here derive from that model; obtain it
  from its own source under its own terms.

## Setup

```bash
pip install -r requirements.txt
```

A few notebooks additionally import project-local modules — `reconstruction_grid` and
`spatiotemporal_sampling` (from the `deep-time-mining` backend) and the parent `pyDTDM`
package — clone those alongside this workflow.

**Paths are templatized.** Machine-specific paths were replaced with placeholders you must
set to your own locations before running:

- `<DATA_ROOT>` — where you unpacked the Zenodo data bundle
- `<PATH_TO>` — parent directory of your `deep-time-mining` / `pyDTDM` checkouts
- `<HOME>`, `<CONDA>` — your home directory / conda prefix

On macOS, if PROJ / Cairo errors appear:

```bash
export PROJ_LIB=$CONDA_PREFIX/share/proj
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

## License

- **Code** — PolyForm Noncommercial License 1.0.0 (see the repository `LICENSE`).
- **Data & figures** — CC BY-NC 4.0.
- **Commercial use** of this code or method is available only through **Geonome Pty Ltd** —
  see [`NOTICE`](NOTICE).

## Citation

If you use this workflow, please cite the paper above and the Zenodo data bundle.
