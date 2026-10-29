from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from test_data import _write_dataset_fixture

_SPEC = importlib.util.spec_from_file_location(
    'check_data', Path(__file__).resolve().parents[1] / 'scripts' / 'check_data.py'
)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


@pytest.fixture
def audit_dataset(tmp_path, monkeypatch):
    info, weak, strong, audio = _write_dataset_fixture(tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {
        'audios_info_csv': str(info), 'weak_labels_csv': str(weak),
        'strong_labels_dir': str(strong), 'original_audio_dir': str(audio),
        'chunk_sec': 20,
    }}))
    args = SimpleNamespace(config=str(config), check_audio=False, decode_audio=True, manifest=None)
    monkeypatch.setattr(checker, 'parse_args', lambda: args)
    return config, weak, args


def test_healthy_dataset_decodes(audit_dataset, capsys):
    assert checker.main() == 0
    assert 'OK:' in capsys.readouterr().out


def test_missing_configured_negatives_fail(audit_dataset, tmp_path, capsys):
    config, _, _ = audit_dataset
    cfg = yaml.safe_load(config.read_text())
    cfg['data']['negative_data_dir'] = str(tmp_path / 'missing')
    config.write_text(yaml.safe_dump(cfg))
    assert checker.main() == 1
    assert 'training would silently lose negatives' in capsys.readouterr().out


@pytest.mark.parametrize('bad_end', [float('inf'), float('nan'), -1, 99])
def test_invalid_or_out_of_bounds_interval_fails(audit_dataset, capsys, bad_end):
    _, weak, _ = audit_dataset
    df = pd.read_csv(weak)
    df.loc[0, 'end_sec'] = bad_end
    df.to_csv(weak, index=False)
    assert checker.main() == 1
    assert 'ERROR:' in capsys.readouterr().out


def test_partial_decode_fails(audit_dataset, monkeypatch, capsys):
    original_load = checker.torchaudio.load

    def truncated(*args, **kwargs):
        waveform, rate = original_load(*args, **kwargs)
        return waveform[..., :-1], rate

    monkeypatch.setattr(checker.torchaudio, 'load', truncated)
    assert checker.main() == 1
    assert 'truncated or inconsistent audio' in capsys.readouterr().out


def test_nonfinite_audio_fails(audit_dataset, monkeypatch, capsys):
    original_load = checker.torchaudio.load

    def nonfinite(*args, **kwargs):
        waveform, rate = original_load(*args, **kwargs)
        waveform[..., 0] = torch.nan
        return waveform, rate

    monkeypatch.setattr(checker.torchaudio, 'load', nonfinite)
    assert checker.main() == 1
    assert 'non-finite samples' in capsys.readouterr().out


def test_download_id_resolves_metadata_title_without_fuzzy_matching(audit_dataset, capsys):
    config, weak, _ = audit_dataset
    cfg = yaml.safe_load(config.read_text())
    info = Path(cfg['data']['audios_info_csv'])
    metadata = pd.read_csv(info)
    title = metadata.loc[0, 'title']
    filename = f'Youtube-abc12345678-{title}.wav'
    audio_dir = Path(cfg['data']['original_audio_dir'])
    (audio_dir / f'{title}.wav').rename(audio_dir / filename)
    strong_dir = Path(cfg['data']['strong_labels_dir'])
    (strong_dir / f'noise_{title}.txt').rename(strong_dir / f'noise_{Path(filename).stem}.txt')
    metadata.loc[0, 'url'] = 'https://www.youtube.com/watch?v=abc12345678'
    metadata.to_csv(info, index=False)
    labels = pd.read_csv(weak)
    labels.loc[0, 'source_file'] = filename
    labels.to_csv(weak, index=False)
    assert checker.main() == 0
    assert 'resolves by download naming convention' in capsys.readouterr().out


def test_missing_original_audio_fails(audit_dataset, capsys):
    config, _, _ = audit_dataset
    cfg = yaml.safe_load(config.read_text())
    next(Path(cfg['data']['original_audio_dir']).glob('Birth*.wav')).unlink()
    assert checker.main() == 1
    assert 'no matching original WAV' in capsys.readouterr().out


@pytest.mark.parametrize('interval', ['3\t1\tclear_approval', '0\t999\tclear_approval'])
def test_bad_strong_interval_fails(audit_dataset, capsys, interval):
    config, _, _ = audit_dataset
    cfg = yaml.safe_load(config.read_text())
    next(Path(cfg['data']['strong_labels_dir']).glob('*.txt')).write_text(interval + '\n')
    assert checker.main() == 1
    assert 'ERROR:' in capsys.readouterr().out


def test_successful_audit_writes_checksum_inventory(audit_dataset, tmp_path):
    import hashlib
    import json

    _, weak, args = audit_dataset
    args.manifest = tmp_path / 'manifest.json'
    assert checker.main() == 0
    inventory = json.loads(args.manifest.read_text())
    entry = next(item for item in inventory['files'] if item['path'] == str(weak.resolve()))
    assert entry['sha256'] == hashlib.sha256(weak.read_bytes()).hexdigest()


def test_decoded_audio_cache_reuses_sources_without_mutating_them(audit_dataset, monkeypatch):
    from crowd_reaction import data as data_module
    config, _, _ = audit_dataset
    cfg = yaml.safe_load(config.read_text())['data']
    splits = data_module.build_split_records(**cfg)
    record = splits.train_records[0]
    original = data_module.load_audio_mono
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(data_module, 'load_audio_mono', counted)
    dataset = data_module.WeakChunkDataset([record, record], audio_cache_size=1)
    a, b = dataset[0], dataset[1]
    assert len(calls) == 1
    assert torch.equal(a['instances'], b['instances'])
