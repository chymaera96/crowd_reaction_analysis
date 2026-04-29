# crowd_reaction_analysis
Base repository for analysis of crowd reactions in political speeches.

This repo includes a PyTorch training package for weakly supervised sound event detection on 20 second speech chunks, using a frozen BEATs encoder and multi-instance learning heads for `crowd`, `approval`, and `disapproval`.

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
- `scripts/infer.py --config configs/default.yaml --checkpoint outputs/exp001/best_val.pt --output-dir outputs/exp001/inference_plots --run-id exp001_infer --wandb-mode online`
- `src/crowd_reaction/data.py` for metadata loading and chunk slicing
- `src/crowd_reaction/model.py` for frozen BEATs + temporal classifier head
- `src/crowd_reaction/eval.py` for weak metrics plus `sed_eval`-based segment and event validation

Dataset inputs:
- `data/audios_info.csv` decides which source files are strong-labeled and therefore validation-only
- `data/weak_labelling/_weak_labels.csv` provides the 20 s weak bags
- `data/negative_data/*.wav` optionally provides additional pre-segmented 20 s `no_crowd` training bags
- `data/strong_labelling/noise_*.txt` provides strong validation intervals
- `data/original_audio_files/*.wav` provides the source audio for the weak/strong metadata-driven records

Known filename inconsistencies in `data/strong_labelling`:
- `noise_Ukip's Douglas Carswell is booed while asking question at PMQs.txt` should be renamed to `noise_Ukip's Douglas Carswell is booed while asking question at PMQs - video.txt` so it matches the corresponding `.wav` filename.
- `noise_Jonas brothers - crowd cheering.txt` should be renamed to `noise_Youtube-82b2ZF3L3XQ-Jonas brothers - crowd cheering.txt` so it matches the naming used by the corresponding source item.
- The Jonas Brothers strong-label entry is also missing its matching audio file in `data/strong_labelling/`, so the `.txt` and `.wav` assets are currently inconsistent.

Internal metadata:
- weak records: `audio_path`, `speech_id`, `chunk_start_sec`, `chunk_end_sec`, structured task targets, `split`
- strong events: `speech_id`, `event_class`, `onset_sec`, `offset_sec`

Weak targets:
- `event`: binary `relevant_event` / crowd detector
- `approval`: independent binary approval detector
- `disapproval`: independent binary disapproval detector
- weak positives are any of `clear_disapproval`, `unclear_disapproval`, `unclear_approval`, `clear_approval`, or `crowd_chorus`
- weak negatives are `no_crowd`
- clear labels train approval/disapproval with full weight; unclear labels train them with `loss.unclear_label_weight`
- `crowd_chorus` trains only `event`
- approval/disapproval are masked for `no_crowd` because they are immaterial when no event is present
- `hard_annotation` is ignored for target construction

Strong validation uses `sed_eval`:
- all strong crowd-like labels collapse to a single positive `crowd` event class
- segment-based metrics at the configured `instance_sec`
- event-based metrics with configurable onset collar and offset ratio

Checkpoint outputs:
- `last.pt` for the final epoch
- `best_val.pt` for the best validation epoch
- both are saved under `<output-dir>/<run-id>/` when `--run-id` is provided

Inference plots:
- `scripts/infer.py` runs the trained model over the strong-labeled validation files
- it saves one PNG per file with a spectrogram background, raw ground-truth strong spans, and prediction score curves for `relevant_event`, `approval`, and `disapproval`
- exported approval/disapproval regions are hard-gated by the `relevant_event` threshold
- if W&B is enabled or `--wandb-mode` / `--run-id` are passed, the saved images are also logged to W&B
