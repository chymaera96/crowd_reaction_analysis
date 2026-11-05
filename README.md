# Crowd reaction analysis

Crowd-event, approval, and disapproval detection for audio recordings. The default
model is the joint CNN–TCN; BEATs and wav2vec2 are also available. This package
supplies crowd detection to the
[speech prosody pipeline](https://github.com/alejandroheca/speech_prosody_analysis).
Users can install it directly without keeping a second repository checkout.
[HANDOVER.md](HANDOVER.md) describes the architectures and training recipes.

## Installation

Use Python **3.13 or 3.14** in your application's environment. Install the wheel
from the [package release](https://github.com/chymaera96/crowd_reaction_analysis/releases/tag/v0.2.0):

```bash
python -m pip install https://github.com/chymaera96/crowd_reaction_analysis/releases/download/v0.2.0/crowd_reaction_analysis-0.2.0-py3-none-any.whl
```

CNN–TCN and BEATs use the base installation. For wav2vec2, add its optional
Transformers dependency:

```bash
python -m pip install "crowd-reaction-analysis[wav2vec2] @ https://github.com/chymaera96/crowd_reaction_analysis/releases/download/v0.2.0/crowd_reaction_analysis-0.2.0-py3-none-any.whl"
```

The package supports PyTorch/TorchAudio 2.8 or newer within the 2.x series.
Use a compatible TorchAudio build with your PyTorch installation. Audio decoding
uses SoundFile; input recordings can be WAV, FLAC, or another format supported
by your SoundFile/libsndfile installation. Audio is mixed to mono and resampled.
For GPU environments, install the PyTorch/TorchAudio builds for the server's
CUDA setup before installing this package.

For development or training from a source checkout:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,wav2vec2,test]"
```

Training and evaluation dependencies are optional and are included by `[train]`.
The Conda recipe in `environment.yml` installs the base inference package.

## Annotate one recording

The installed command works from any directory. It automatically downloads the
CNN–TCN checkpoint the first time it is needed:

```bash
crowd-reaction --audio /path/to/speech.wav --output-dir annotations/example \
  --mode event --label-scores --min-mean-score 0.8
```

This detects any crowd reaction, uses the label `crowd`, appends each region's
mean probability, and retains regions whose mean probability exceeds 0.8.
For approval/disapproval annotations, use `--mode polarity` instead.

The command writes:

- `predicted_segments.csv`: headerless Sonic Visualiser region annotations.
- `scores.json`: temporal probability scores.
- `plot.png`: spectrogram, predicted regions, and score functions.

Load the original recording in Sonic Visualiser and import the CSV as a region
annotation layer. Its columns are `start_seconds,duration_seconds,label`:

```csv
12.400000,2.800000,crowd mean=0.91
31.200000,1.600000,crowd mean=0.87
```

The first region runs from 12.4 to 15.2 seconds. The second column is duration.
Omit `--label-scores` for plain labels.

Useful options:

| Option | Effect |
|---|---|
| `--model cnn_tcn`, `beats`, or `wav2vec2` | Select the model; default is `cnn_tcn` |
| `--device cpu` or `cuda` | Inference device; default is CPU |
| `--batch-size 1` | Number of audio chunks processed together; default is 1 |
| `--event-threshold`, `--attribute-threshold` | Override the model's probability thresholds |
| `--median-filter-sec 3` | Segment smoothing window; use `0` to disable |
| `--min-mean-score 0.8` | Retain segments with mean probability above 0.8 |
| `--annotations labels.txt` | Overlay tab-separated onset/offset/label ground truth on the plot |
| `--no-score-functions` | Hide probability curves and threshold lines |
| `--no-progress` | Hide the batch progress bar |
| `--cache-dir /path/to/cache` | Choose a model-weight cache directory |
| `--checkpoint /path/to/model.pt` | Use local weights, including offline |
| `--config /path/to/config.yaml` | Supply a custom configuration |

BEATs or wav2vec2 use the same command with `--model beats` or
`--model wav2vec2`; no configuration paths are needed for the released models.

## Python API and prosody integration

Create one detector and reuse it across recordings so the model is loaded once:

```python
from crowd_reaction import CrowdDetector

detector = CrowdDetector(model="cnn_tcn", device="cpu")
result = detector.predict(
    "speech.wav", mode="event", min_mean_score=0.8, show_progress=False,
)
result.save("annotations/example", label_scores=True)
```

`result` exposes `predicted_regions` as `(onset, offset, label)` tuples, together
with `scores`, `label_names`, `times_sec`, and `instance_sec`. Event scores use
the internal name `relevant_event`; CSV export labels them `crowd`.

The speech-prosody pipeline can use the detector directly and obtain the same
start/end/mean-score tuples it currently consumes:

```python
regions = [
    (onset, offset, result.region_mean_score(onset, offset, label))
    for onset, offset, label in result.predicted_regions
]
```

Install the package in the environment running the crowd-detection step, then
replace the neighbouring-repository import with `from crowd_reaction import
CrowdDetector`. Repository paths such as `CROWD_CNN_REPO` are unnecessary with
this API. `result.save(..., plot=False)` writes only CSV and JSON. Call
`run_audio_inference` from `crowd_reaction` for one-shot inference; use
`CrowdDetector` when processing multiple recordings.

## Models, downloads, and offline use

| Model | Output rate | Release checkpoint |
|---|---|---|
| CNN–TCN, joint | 25 Hz | `cnn_tcn.pt` |
| BEATs | 1 Hz | `beats.pt` |
| wav2vec2, layer 3 | 2 Hz | `wav2vec2.pt` |

Weights remain in the [v0.1.0 model release](https://github.com/chymaera96/crowd_reaction_analysis/releases/tag/v0.1.0).
The package contains configurations and SHA-256 checksums, downloads only the
selected model, and verifies existing and downloaded files. Weights are cached
per user and model release: `~/Library/Caches/crowd_reaction/v0.1.0` on macOS,
`~/.cache/crowd_reaction/v0.1.0` on Linux (or under `XDG_CACHE_HOME`), and
`%LOCALAPPDATA%/crowd_reaction/v0.1.0` on Windows. Set `CROWD_REACTION_CACHE`
to override the cache root, or pass `cache_dir` / `--cache-dir` to use an exact
directory. Existing valid checkpoints are reused. Failed downloads never replace
an existing checkpoint.

Prefetch weights on a machine with internet access:

```bash
crowd-reaction-download --model cnn_tcn --output-dir ./weights
```

Select several models with `--model cnn_tcn beats wav2vec2`. Use `--force` to
replace a different existing checkpoint after verifying its replacement.
For offline inference, pass the downloaded file:

```python
detector = CrowdDetector(checkpoint_path="weights/cnn_tcn.pt")
```

Each released checkpoint contains the encoder weights and architecture metadata;
inference with an explicit checkpoint needs no network access. Custom trained
models can use both `checkpoint_path` and `config_path`.

## Existing source-checkout commands

The original script entry points remain available from a checkout:

```bash
python scripts/api.py --audio /path/to/speech.wav --output-dir annotations/example --mode event
python scripts/download_weights.py --model cnn_tcn
```

The legacy download script saves to the checkout's `weights/` directory;
the installed download command uses the shared cache unless `--output-dir` is
specified. Existing calls supplying `--config` and `--checkpoint` still work.
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
plots and overall validation metrics for BEATs, wav2vec2, and the best joint
CNN–TCN are in `outputs/<model>/`; the combined metric table is
`outputs/validation_metrics.csv`. The CNN–TCN folder also includes the MLK plot.

## Tests

The installed v0.2.0 wheel has been tested on an NVIDIA A100 80 GB with Python
3.14.10, PyTorch 2.15.0, TorchAudio 2.11.1, and CUDA 13.2. CNN–TCN inference ran
on CUDA with only the base dependencies installed. The Python API and installed
CLI both generated annotation CSVs, score JSON, and plots. Full-recording
inference on the 1,013-second MLK speech took about 9.5 seconds, excluding plot
generation, with batch size 4 after the initial excerpt test.

```bash
python -m pytest -q
```

## Citation

Hermida Carrillo, A. C., Bhattacharjee, A., & Benetos, E. (2026) [To the sound of your
applause: a signal processing approach to crowd-speaker communication in political
speeches.](https://spiral.imperial.ac.uk/entities/publication/29bbc09e-718a-4190-a838-fa276b371e9a) *UK and Ireland (UKIS) Workshop.*
