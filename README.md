# Ablation tables as measuring instruments — code, results and paper source

Everything needed to check the claims in the paper: the code, the recorded numbers behind every
table, and the paper's LaTeX source. This bundle is exported from the project's working repository
from an explicit manifest, so it contains only what is listed below.

| | |
|---|---|
| `paper/` | The paper's LaTeX source, bibliography, figures and the scripts that generate them |
| `scripts/lrrnet/` | Patches, capacity verifiers, the fusion and evaluation pipeline, and one analysis script per experiment |
| `scripts/crossfuse/` | The CrossFuse positive control, the untrained-network mechanism, the passthrough baselines and the σ-estimator audit |
| `scripts/cddfuse/` | The CDDFuse test of the residual-copy mechanism, with its log and summary |
| `results/` | The numbers of record, with per-run accounting |
| `data/` | The recorded metrics of every experiment, per run, per image where recorded (index: `data/INDEX.md`) |
| `registrations/` | The pre-registration of each confirmatory run, written before it |
| `notebooks/` | The Kaggle cells each run was launched from |
| `MANIFEST.sha256` | Checksums for every file above |

## What is here, and what is not

**Not here:** trained checkpoints, image corpora, and PDFs. The build refuses to include them, and
refuses any file over 2 MB or any line that looks like a credential. The datasets are public
(TNO, RoadScene, MSRS, LLVIP, M3FD) and are named with their sources in the paper.

**Deviations are recorded, not hidden.** LRRNet's published training corpus cannot be rebuilt from
what its authors released, so LLVIP is substituted; the loss network uses standard ImageNet VGG16
weights, which that paper does not ship; and seeding was added to code that contains none. Numbers
here describe that architecture and procedure on that data and are never compared against the
authors' printed values.

## Upstream code, pinned

The methods studied are the authors' own released code, cloned at these commits (each is the repository's
head, dated before every run reported here, so it is the version the runs used):

| Method | Repository | Commit |
|---|---|---|
| LRRNet | `hli1221/imagefusion-LRRNet` | `e489fc325ea6acb9b7327b1de2c0cc4a09172ba4` (2023-08-18) |
| CrossFuse | `CidanShi/CrossFuse` | `d9cf1b6f8c4159d0dfa4464140a36a69c1a7e043` (2025-04-02) |
| CDDFuse | `Zhaozixiang1228/MMIF-CDDFuse` | `73681d28f6100461f1b214ecf77a93b495be227b` (2026-06-08) |

The notebook cells in `notebooks/` clone LRRNet without naming a commit, as they were run; check out the
commit above after cloning to rebuild the same code.

**What re-running needs.** Trained checkpoints and image data are not included. The datasets are public
(TNO, RoadScene, MSRS, LLVIP, M3FD); training needs a GPU (each LRRNet run is about one GPU-hour on a T4,
each CrossFuse run about three). The recorded metrics of every experiment are in `data/` (index: `data/INDEX.md`) and
each confirmatory run's pre-registration in `registrations/`, so the numbers can be checked without
training. Not here: checkpoints, fused images, datasets, and the exact versions of packages other
than torch, which were not recorded.

## Reproducing a result

Each analysis script in `scripts/lrrnet/` carries, in its own source, **the pre-registered reading
of both possible outcomes** — written before the run it analyses. That is the point of the design:
an interpretation chosen after seeing the numbers is excluded by construction.

`check_tied_capacity.py` and `check_numblock_capacity.py` are the cheapest starting point: they run
on a laptop in seconds, need no data, and verify the parameter-count claims the paper's constructive
section rests on.

## The paper source

```
python paper/check_paper.py         # structural checks on paper.tex; exit 1 on any FAIL
python paper/make_paper_figures.py  # regenerate every figure from the recorded numbers
```

`paper.tex` is itself generated, from Markdown section drafts that are not part of this bundle, so
the generator is not shipped either — a script that cannot run is worse than no script. The figure
generator *is* shipped and does run: every number it plots is transcribed in its source, with the
results file each came from named in a comment above it.

## Licence

The original code in this repository is released under the MIT licence (`LICENSE`). Patches to
third-party code (LRRNet, CrossFuse and CDDFuse) apply to those projects'
own sources, which remain under their authors' licences and are not redistributed here.
