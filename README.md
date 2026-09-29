# Dual-WM

**BEYOND A SINGLE LATENT SPACE: A DUAL-LATENT WORLD MODEL FOR LONG-HORIZON PLANNING**

This repository provides the core Dual-WM implementation for use and reuse, including the main training and planning pipelines.

## Training

```bash
export STABLEWM_HOME=/absolute/path/to/training-data-root
task=tworoom
python -m scripts.train.dual_wm -cn "$task" stage=low
python -m scripts.train.dual_wm -cn "$task" stage=high \
  init_from="$STABLEWM_HOME/checkpoints/low_${task}/weights_final.pt"
```

Set `task` to `tworoom`, `reacher`, `pusht`, `cube`, or `sokobanlong`. Training datasets go under `$STABLEWM_HOME/datasets`.

## Offset-100 evaluation

```bash
export STABLEWM_HOME=/absolute/path/to/dual-wm-publication/data
export DUALWM_CHECKPOINT_DIR=/absolute/path/to/dual-wm-publication/checkpoints
python -m scripts.plan.eval_wm -cn tworoom
python -m scripts.plan.eval_wm -cn reacher
python -m scripts.plan.eval_wm -cn pusht
python -m scripts.plan.eval_wm -cn cube
python -m scripts.plan.eval_wm -cn sokobanlong
```

## Note

It is not a complete archive of all paper experiments. Ablation experiments, baseline implementations, and paper-specific analysis scripts are outside the scope of this release.
