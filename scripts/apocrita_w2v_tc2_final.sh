#!/usr/bin/env bash
# Invoke with srun inside an existing GPU allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SLURM_JOB_ID:?Run inside the allocated GPU session}"
: "${CROWD_PYTHON:?Set the project Python executable}"
: "${CROWD_DATA_ROOT:?Set the dataset directory}"
: "${CROWD_OUTPUT_ROOT:?Set the output directory}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
RUN_DIR="$CROWD_OUTPUT_ROOT/w2v_tc2_final_layer12"
mkdir "$RUN_DIR"
export RUN_DIR
"$CROWD_PYTHON" -u - <<'PY'
import os
from pathlib import Path
import torch
import yaml
from transformers import Wav2Vec2Model
assert torch.cuda.is_available(), "GPU required"
config = yaml.safe_load(Path('configs/wav2vec2_tc2_final.yaml').read_text())
assert config['model']['wav2vec2_layer_index'] == 12
assert 'wav2vec2_layer_indices' not in config['model']
encoder, loading = Wav2Vec2Model.from_pretrained(
    config['model']['wav2vec2_model_name'], output_loading_info=True,
)
assert not loading['missing_keys'], f"Missing pretrained weights: {loading['missing_keys']}"
assert config['model']['wav2vec2_layer_index'] == encoder.config.num_hidden_layers
del encoder
for key, value in config['data'].items():
    if isinstance(value, str) and value.startswith('data/'):
        config['data'][key] = str(Path(os.environ['CROWD_DATA_ROOT']) / value.removeprefix('data/'))
Path(os.environ['RUN_DIR'], 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
print('GPU:', torch.cuda.get_device_name())
print('All pretrained wav2vec2 encoder weights loaded successfully.')
print('wav2vec2: final transformer layer 12; tc2 settings; augmentation disabled', flush=True)
PY
git rev-parse HEAD > "$RUN_DIR/source-commit.txt"
"$CROWD_PYTHON" -u scripts/check_data.py --config "$RUN_DIR/config.yaml" --check-audio
"$CROWD_PYTHON" -u scripts/train.py --config "$RUN_DIR/config.yaml" \
  --wav2vec2-layer 12 --output-dir "$RUN_DIR" --wandb-mode disabled
