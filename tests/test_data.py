from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio

from crowd_reaction.data import WeakChunkDataset, slice_waveform, split_into_instances


def test_slice_waveform_pads_to_chunk_length() -> None:
    waveform = torch.arange(10, dtype=torch.float32)
    chunk = slice_waveform(waveform, start_sec=0.0, end_sec=2.0, sample_rate=4, target_num_samples=8)
    assert chunk.shape == (8,)
    assert torch.allclose(chunk[:8], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32))


def test_split_into_instances_exact_second_bins() -> None:
    chunk = torch.arange(12, dtype=torch.float32)
    instances = split_into_instances(chunk, sample_rate=2, instance_sec=1.0, chunk_sec=6.0)
    assert instances.shape == (6, 2)
    assert torch.equal(instances[0], torch.tensor([0.0, 1.0]))
    assert torch.equal(instances[-1], torch.tensor([10.0, 11.0]))


def test_dataset_reads_metadata_and_returns_expected_shapes(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    waveform = torch.linspace(-1.0, 1.0, 16000 * 4).unsqueeze(0)
    torchaudio.save(str(audio_path), waveform, sample_rate=16000)

    df = pd.DataFrame(
        [
            {
                "audio_path": str(audio_path),
                "speech_id": "speech-1",
                "chunk_start_sec": 1.0,
                "chunk_end_sec": 3.0,
                "label_0": 1,
                "label_1": 0,
            }
        ]
    )
    metadata_path = tmp_path / "weak.csv"
    df.to_csv(metadata_path, index=False)

    dataset = WeakChunkDataset(
        str(metadata_path),
        sample_rate=16000,
        chunk_sec=2.0,
        instance_sec=1.0,
        num_classes=2,
    )
    item = dataset[0]
    assert item["waveform"].shape == (32000,)
    assert item["instances"].shape == (2, 16000)
    assert torch.equal(item["labels"], torch.tensor([1.0, 0.0]))
