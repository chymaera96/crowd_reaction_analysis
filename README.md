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
- pass `--wandb-mode online` or `--wandb-mode offline`
- pass `--run-id your_run_name` to name both the W&B run and the checkpoint subdirectory

Main entrypoints:
- `scripts/train.py --config configs/default.yaml --output-dir /path/to/output --run-id exp001 --wandb-mode online`
- `src/crowd_reaction/data.py` for metadata loading and chunk slicing
- `src/crowd_reaction/model.py` for frozen BEATs + temporal classifier head
- `src/crowd_reaction/eval.py` for weak metrics plus `sed_eval`-based segment and event validation

Dataset inputs:
- `data/audios_info.csv` decides which source files are strong-labeled and therefore validation-only
- `data/weak_labelling/_weak_labels.csv` provides the 20 s weak bags
- `data/strong_labelling/noise_*.txt` provides strong validation intervals
- `data/original_audio_files/*.wav` are the only source audio files used for training and validation

Known filename inconsistencies in `data/strong_labelling`:
- `noise_Ukip's Douglas Carswell is booed while asking question at PMQs.txt` should be renamed to `noise_Ukip's Douglas Carswell is booed while asking question at PMQs - video.txt` so it matches the corresponding `.wav` filename.
- `noise_Jonas brothers - crowd cheering.txt` should be renamed to `noise_Youtube-82b2ZF3L3XQ-Jonas brothers - crowd cheering.txt` so it matches the naming used by the corresponding source item.
- The Jonas Brothers strong-label entry is also missing its matching audio file in `data/strong_labelling/`, so the `.txt` and `.wav` assets are currently inconsistent.

Internal metadata:
- weak records: `audio_path`, `speech_id`, `chunk_start_sec`, `chunk_end_sec`, `label_0`, `label_1`, `split`
- strong events: `speech_id`, `event_class`, `onset_sec`, `offset_sec`

Weak targets:
- target `0`: any disapproval (`clear_disapproval` or `unclear_disapproval`)
- target `1`: any approval (`clear_approval` or `unclear_approval`)

Strong validation uses `sed_eval`:
- segment-based metrics at the configured `instance_sec`
- event-based metrics with configurable onset collar and offset ratio

Checkpoint outputs:
- `last.pt` for the final epoch
- `best_val.pt` for the best validation epoch
- both are saved under `<output-dir>/<run-id>/` when `--run-id` is provided
