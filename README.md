# crowd_reaction_analysis

Sound event detection for crowd reactions in political speech audio. The public usage path is `scripts/api.py`, which runs a trained checkpoint on one audio file and writes score functions, predicted segments, and a plot.

## Installation

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate crowd-reaction-analysis
pip install -e .
```

## Pretrained Models

The repo supports two frozen audio encoders:

- BEATs: `configs/default.yaml`
- wav2vec2: `configs/wav2vec2.yaml`

BEATs uses a local checkpoint configured by:

```yaml
model:
  encoder_type: beats
  beats_checkpoint_path: /path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

Update `model.beats_checkpoint_path` before using the BEATs config on a new machine.
The BEATs checkpoint can be downloaded from [here](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea).

wav2vec2 uses Hugging Face Transformers:

```yaml
model:
  encoder_type: wav2vec2
  wav2vec2_model_name: facebook/wav2vec2-base
```

The wav2vec2 weights are downloaded automatically by `transformers` and cached in the Hugging Face cache. If your machine cannot write to the default cache, set a writable cache location before running:

```bash
export HF_HOME=/path/to/writable/huggingface_cache
```

Both configs expect 16 kHz mono audio. BEATs currently uses 1 s output bins, while wav2vec2 uses 0.5 s output bins.

## API Usage

Run one audio file through a trained checkpoint:

```bash
python scripts/api.py \
  --config configs/wav2vec2.yaml \
  --checkpoint outputs/w2v_tc1_alt/best_segment_f1.pt \
  --audio /path/to/input.wav \
  --output-dir api_outputs/example \
  --mode polarity \
  --batch-size 1
```

The API writes:

- `scores.json`: score functions as JSON lists
- `predicted_segments.csv`: Sonic Visualiser-style rows, `start_sec,duration,label`
- `plot.png`: spectrogram with prediction spans and optional score functions

Modes:

- `--mode polarity`: default mode. Exports `approval` and `disapproval` predicted spans. The approval/disapproval score functions are event-conditioned.
- `--mode event`: exports only `relevant_event` predicted spans, with no approval/disapproval assignment.

Useful CLI options:

```bash
--event-threshold 0.6
--attribute-threshold 0.5
--median-filter-sec 1.5
--no-score-functions
--no-progress
--device cpu
--device cuda
--device mps
```


Python usage:

```python
from scripts.api import (
    plot_inference_result,
    run_audio_inference,
    write_predicted_segments_csv,
    write_scores_json,
)

result = run_audio_inference(
    audio_path="/path/to/input.wav",
    config_path="configs/wav2vec2.yaml",
    checkpoint_path="outputs/w2v_tc1_alt/best_segment_f1.pt",
    mode="polarity",
    batch_size=1,
)

write_scores_json(result, "api_outputs/example/scores.json")
write_predicted_segments_csv(result, "api_outputs/example/predicted_segments.csv")
plot_inference_result(result, "api_outputs/example/plot.png")
```

The returned `result` contains:

- `label_names`
- `times_sec`
- `scores`
- `predicted_regions`
- thresholds and metadata used for the run

You can edit `result.scores` or `result.predicted_regions` before plotting if you want custom post-processing.

## Citation

Citation details will be available soon.
