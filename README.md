# Dual-WM

**BEYOND A SINGLE LATENT SPACE: A DUAL-LATENT WORLD MODEL FOR LONG-HORIZON PLANNING**

This repository provides the core Dual-WM implementation for use and reuse, including the main training and planning pipelines.

## Pretrained checkpoints

The five task checkpoints are available on Hugging Face: [Delin1001/Dual-WM-Checkpoints](https://huggingface.co/Delin1001/Dual-WM-Checkpoints).

Run the following from this repository's root directory to download and place them at the default paths:

```bash
curl -L --fail --retry 3 \
  "https://huggingface.co/Delin1001/Dual-WM-Checkpoints/resolve/main/dual-wm-checkpoints.tar.gz?download=true" \
  -o dual-wm-checkpoints.tar.gz
mkdir -p checkpoints
tar -xzf dual-wm-checkpoints.tar.gz -C checkpoints --strip-components=1
```

The archive contains a top-level `dual-wm-checkpoints/` directory. `--strip-components=1` extracts its contents directly into `checkpoints/`:

```text
checkpoints/
├── TwoRoomCPT/TwoRoomCPT.pt
├── ReacherCPT/ReacherCPT.pt
├── PushTCPT/PushTCPT.pt
├── CubeCPT/CubeCPT.pt
└── SokobanLongCPT/SokobanLongCPT.pt
```

Each task directory also includes `config.json` and an `evaluation.yaml` configuration snapshot. Use this repository's `scripts/plan/config/` files for evaluation with the current default paths. With the required datasets under `datasets/`, run `python -m scripts.plan.eval_wm -cn tworoom` from the repository root. If set, `STABLEWM_HOME` and `DUALWM_CHECKPOINT_DIR` override the default paths. Datasets are not included in the checkpoint archive.

Archive SHA-256: `57bcf47d41eb27c650ec6c27515354fbc3956133c7d5b8d81f6426bf8acf6ebf`.

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
