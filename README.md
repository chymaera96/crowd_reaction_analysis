# Crowd reaction analysis

Sound event detection for crowd presence, approval, and disapproval in speech audio.
The primary architectures are BEATs, wav2vec2, and CNN–TCN.
[HANDOVER.md](HANDOVER.md) describes their architectures and training recipes.

## Installation

Use Python 3.11 or 3.12. Run commands from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

A Conda environment is also provided:

```bash
conda env create -f environment.yml
conda activate crowd-reaction-analysis
python -m pip install -e ".[test]"
```

## Models and weights

Place the supplied classifier checkpoints in `weights/`:

| Model | Configuration | Local weights | Output rate |
|---|---|---|---|
| BEATs | `configs/beats.yaml` | `weights/beats.pt` | 1 Hz |
| wav2vec2 | `configs/wav2vec2.yaml` | `weights/wav2vec2.pt` | 2 Hz |
| CNN–TCN, joint | `configs/cnn_tcn.yaml` | `weights/cnn_tcn.pt` | 25 Hz |

The checkpoints include their encoder configuration and weights. Inference loads
locally. The `weights/` directory is distributed separately from Git.

## Inference

Run the joint CNN–TCN model on an audio file:

```bash
python scripts/api.py \
  --config configs/cnn_tcn.yaml \
  --checkpoint weights/cnn_tcn.pt \
  --audio /path/to/audio.wav \
  --output-dir annotations/example \
  --mode polarity --batch-size 1 --device cpu
```

For BEATs or wav2vec2, replace the configuration and weight filenames with the
corresponding pair from the table. Audio is resampled and mixed to mono.
Use `--device cuda` for GPU inference.

The API writes:

- `predicted_segments.csv`: headerless Sonic Visualiser rows, `start_seconds,duration_seconds,label`.
- `scores.json`: temporal probability scores.
- `plot.png`: spectrogram, predictions, and score functions.

`--mode polarity` exports approval/disapproval segments. `--mode event` exports
`relevant_event` segments. Override thresholds with `--event-threshold` and
`--attribute-threshold`. `--median-filter-sec` controls segment smoothing (default
3 seconds); use `0` to retain raw thresholded boundaries. `--no-score-functions`
hides score curves. Add `--annotations /path/to/labels.txt` to overlay ground truth.

The saved MLK speech annotations are in `annotations/mlk.csv`.

## Training data

Arrange the dataset as follows:

```text
data/
  audios_info.csv
  original_audio_files/*.wav
  weak_labelling/_weak_labels.csv
  strong_labelling/*.txt
  negative_data/*.wav
```

`audios_info.csv` contains `title` and `strong_label`. Weak labels contain
`source_file`, `start_sec`, `end_sec`, `clear_approval`, `unclear_approval`,
`clear_disapproval`, `unclear_disapproval`, `crowd_chorus`, and `no_crowd`.
Strong TXT files contain tab-separated `onset_seconds`, `offset_seconds`, and
label. Sources with strong annotations are reserved for validation. Update
`data` paths in the configuration when using another directory layout.

## Training and validation

```bash
python scripts/check_data.py --config configs/cnn_tcn.yaml --decode-audio
python scripts/train.py \
  --config configs/cnn_tcn.yaml --output-dir training/cnn_tcn --wandb-mode disabled
python scripts/results.py \
  --config configs/cnn_tcn.yaml \
  --checkpoint training/cnn_tcn/best_polarity_segment_f1.pt \
  --output outputs/cnn_tcn/metrics.json
```

Use `configs/beats.yaml` or `configs/wav2vec2.yaml` to train the frozen-encoder
classifiers. BEATs initializes its frozen encoder from `weights/beats.pt`.
wav2vec2 training downloads `facebook/wav2vec2-base` through Transformers.
Training uses CUDA when available and CPU otherwise.

Use `configs/cnn_tcn_event.yaml` to train an event-only model. Its segment-F1
checkpoint is `best_event_segment_f1.pt`. Joint models save
`best_polarity_segment_f1.pt`, `best_polarity_event_f1.pt`,
`best_weak_polarity_auprc.pt`, `best_relevant_event_f1.pt`, and `last.pt`.

Generate annotated validation plots with:

```bash
python scripts/infer.py \
  --config configs/cnn_tcn.yaml --checkpoint weights/cnn_tcn.pt \
  --output-dir annotations/validation --wandb-mode disabled
```

This generates plots and exports predicted and ground-truth segment CSVs. Saved
overall validation metrics for BEATs, wav2vec2, and the joint CNN–TCN are in
`outputs/<model>/metrics.json`; the combined table is
`outputs/validation_metrics.csv`. Generated plots are excluded from Git.

## Tests

```bash
python -m pytest -q
```

## Citation

Hermida Carrillo, A. C., Bhattacharjee, A., & Benetos, E. *To the sound of your
applause: a signal processing approach to crowd-speaker communication in political
speeches.* UK and Ireland (UKIS) Workshop.
