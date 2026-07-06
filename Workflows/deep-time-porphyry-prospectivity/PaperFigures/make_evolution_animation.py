# -*- coding: utf-8 -*-
"""Render the two deep-time evolution animations through 170-0 Ma.
  Animation A: crustal thickness | subducted carbonate (+ plate motion)
  Animation B: slab sediment + convergence | spatiotemporal prospectivity
Frames are written to temp dirs, assembled to MP4 + GIF, then removed.
Run:  python make_evolution_animation.py [step_Ma]   (default 2)
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import fig_evolution2 as E

OUT = os.path.join(os.path.dirname(__file__), "out")
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TIMES = list(range(170, -1, -STEP))
# prospectivity per frame (for Figure B) — extracted at matching steps
ANIM = os.path.join(os.path.dirname(__file__), "data", "anim_grid.parquet")
prosp = pd.read_parquet(ANIM) if os.path.exists(ANIM) else None

def _render(make, tag, prosp_df=None):
    fr = f"{OUT}/_frames_{tag}"; os.makedirs(fr, exist_ok=True); paths = []
    for i, t in enumerate(TIMES):
        fig = make(t) if prosp_df is None else make(t, prosp_df=prosp_df)
        p = f"{fr}/f_{t:04d}.png"; fig.savefig(p, dpi=105); plt.close(fig); paths.append(p)
        if i % 15 == 0:
            print(f"  [{tag}] {i+1}/{len(TIMES)} ({t} Ma)", flush=True)
    imgs = [imageio.imread(p) for p in paths]
    h = min(im.shape[0] for im in imgs); w = min(im.shape[1] for im in imgs)
    h -= h % 2; w -= w % 2          # libx264 requires even dimensions
    imgs = [im[:h, :w] for im in imgs]
    imageio.mimsave(f"{OUT}/Animation_{tag}.mp4", imgs, fps=7, quality=8, macro_block_size=None)
    imageio.mimsave(f"{OUT}/Animation_{tag}.gif", imgs, duration=0.16, loop=0)
    for p in paths:
        os.remove(p)
    os.rmdir(fr)
    print("WROTE", f"{OUT}/Animation_{tag}.mp4 (+ .gif)", flush=True)

if __name__ == "__main__":
    _render(E.make_figA, "A_crustal_carbonate")
    _render(E.make_figB, "B_sediment_convergence_prospectivity", prosp_df=prosp)
    print("DONE both animations")
