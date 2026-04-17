# crowd_reaction_analysis
Base repository for analysis of crowd reactions in political speeches.

This repo includes a PyTorch training package for weakly supervised sound event detection on 20 second speech chunks, using a frozen BEATs encoder and a multi-instance learning head for two independent crowd-reaction labels.

## Setup

Conda setup:
- `conda env create -f environment.yml`
- `conda activate crowd-reaction-analysis`
- `pip install -e .`

If you want Weights & Biases logging:
- `wandb login`
- set `wandb.enabled: true` in the config
- optionally set `wandb.mode: offline` for local-only experiment tracking

Main entrypoints:
- `scripts/train.py --config configs/default.yaml --output-dir /path/to/output`
- `src/crowd_reaction/data.py` for metadata loading and chunk slicing
- `src/crowd_reaction/model.py` for frozen BEATs + temporal classifier head
- `src/crowd_reaction/eval.py` for weak metrics plus `sed_eval`-based segment and event validation

Dataset inputs:
- `data/audios_info.csv` decides which source files are strong-labeled and therefore validation-only
- `data/weak_labelling/_weak_labels.csv` provides the 20 s weak bags
- `data/strong_labelling/noise_*.txt` provides strong validation intervals
- `data/original_audio_files/*.wav` are the only source audio files used for training and validation

Internal metadata:
- weak records: `audio_path`, `speech_id`, `chunk_start_sec`, `chunk_end_sec`, `label_0`, `label_1`, `split`
- strong events: `speech_id`, `event_class`, `onset_sec`, `offset_sec`

Weak targets:
- target `0`: any disapproval (`clear_disapproval` or `unclear_disapproval`)
- target `1`: any approval (`clear_approval` or `unclear_approval`)

Strong validation uses `sed_eval`:
- segment-based metrics at the configured `instance_sec`
- event-based metrics with configurable onset collar and offset ratio
