from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import torchaudio

from crowd_reaction.data import (
    WeakChunkDataset,
    WeakChunkRecord,
    apply_audio_augmentation,
    build_audio_augmentation,
    build_split_records,
    build_strong_validation_split,
    collate_batch,
    normalize_name,
    parse_strong_label_file,
    slice_waveform,
    split_into_instances,
    weak_row_to_targets,
)


def _write_audio(path: Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate=sample_rate)


def _write_dataset_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    audio_dir = root / "data" / "original_audio_files"
    weak_dir = root / "data" / "weak_labelling"
    strong_dir = root / "data" / "strong_labelling"
    info_path = root / "data" / "audios_info.csv"
    weak_csv_path = weak_dir / "_weak_labels.csv"

    _write_audio(audio_dir / "Birth control question booed at CNN Arizona debate.wav", torch.linspace(0.0, 1.0, 16000 * 25))
    _write_audio(audio_dir / "Taylor Swift booed at Super Bowl #taylorswift #superbowl ｜ Sports Illustrated.wav", torch.ones(16000 * 20))
    _write_audio(weak_dir / "noise_Birth control question booed at CNN Arizona debate.wav", torch.zeros(16000 * 20))
    _write_audio(strong_dir / "noise_Birth control question booed at CNN Arizona debate.wav", torch.zeros(16000 * 20))

    pd.DataFrame(
        [
            {"title": "Birth control question booed at CNN Arizona debate", "strong_label": 1},
            {"title": "Taylor Swift booed at Super Bowl #taylorswift #superbowl | Sports Illustrated", "strong_label": ""},
        ]
    ).to_csv(info_path, index=False)

    pd.DataFrame(
        [
            {
                "source_file": "noise_Birth control question booed at CNN Arizona debate.wav",
                "start_sec": 0.0,
                "end_sec": 20.0,
                "clear_disapproval": 1,
                "unclear_disapproval": 0,
                "unclear_approval": 0,
                "clear_approval": 0,
                "hard_annotation": 1,
                "no_crowd": 0,
                "crowd_chorus": 0,
            },
            {
                "source_file": "noise_Taylor Swift booed at Super Bowl #taylorswift #superbowl | Sports Illustrated.wav",
                "start_sec": 0.0,
                "end_sec": 20.0,
                "clear_disapproval": 0,
                "unclear_disapproval": 0,
                "unclear_approval": 1,
                "clear_approval": 0,
                "hard_annotation": 0,
                "no_crowd": 1,
                "crowd_chorus": 0,
            },
        ]
    ).to_csv(weak_csv_path, index=False)

    (strong_dir / "noise_Birth control question booed at CNN Arizona debate.txt").write_text(
        "1.0\t3.0\tclear_disapproval\n5.0\t6.5\tcrowd_chorus\n",
        encoding="utf-8",
    )
    return info_path, weak_csv_path, strong_dir, audio_dir


def test_normalize_name_reconciles_unicode_and_noise_prefix() -> None:
    assert normalize_name("noise_Taylor Swift booed at Super Bowl #taylorswift #superbowl | Sports Illustrated.wav") == normalize_name(
        "Taylor Swift booed at Super Bowl #taylorswift #superbowl ｜ Sports Illustrated.wav"
    )
    assert normalize_name("'Why are you booing me? I can't see?' üò≥ #ufc321") == normalize_name(
        "'Why are you booing me？ I can't see？' 😳 #ufc321.wav"
    )
    assert normalize_name("noise_'Why are you booing me？ I can't see？' 😳 #ufc321_seg001.wav") == normalize_name(
        "'Why are you booing me? I can't see?' #ufc321.wav"
    )


def test_weak_row_to_targets_maps_clear_disapproval() -> None:
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 1,
                "unclear_disapproval": 0,
                "unclear_approval": 0,
                "clear_approval": 0,
                "hard_annotation": 1,
                "no_crowd": 0,
                "crowd_chorus": 0,
            }
        )
    )
    assert targets.event_target == (1.0,)
    assert targets.event_mask == 1.0
    assert targets.approval_target == (0.0,)
    assert targets.approval_mask == 1.0
    assert targets.disapproval_target == (1.0,)
    assert targets.disapproval_mask == 1.0


def test_weak_row_to_targets_treats_crowd_chorus_as_approval() -> None:
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 0,
                "unclear_disapproval": 0,
                "unclear_approval": 0,
                "clear_approval": 0,
                "hard_annotation": 0,
                "no_crowd": 0,
                "crowd_chorus": 1,
            }
        )
    )
    assert targets.event_target == (1.0,)
    assert targets.event_mask == 1.0
    assert targets.approval_target == (1.0,)
    assert targets.approval_mask == 1.0
    assert targets.disapproval_target == (0.0,)
    assert targets.disapproval_mask == 1.0


def test_weak_row_to_targets_masks_contradictory_no_crowd_and_crowd_labels() -> None:
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 1,
                "unclear_disapproval": 1,
                "unclear_approval": 0,
                "clear_approval": 0,
                "hard_annotation": 0,
                "no_crowd": 1,
                "crowd_chorus": 0,
            }
        )
    )
    assert targets.event_mask == 0.0
    assert targets.approval_mask == 0.0
    assert targets.disapproval_target == (1.0,)
    assert targets.disapproval_mask == 0.0


def test_weak_row_to_targets_allows_approval_and_disapproval_together() -> None:
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 1,
                "unclear_disapproval": 0,
                "unclear_approval": 1,
                "clear_approval": 0,
                "hard_annotation": 0,
                "no_crowd": 0,
                "crowd_chorus": 0,
            }
        )
    )
    assert targets.event_target == (1.0,)
    assert targets.event_mask == 1.0
    assert targets.approval_target == (1.0,)
    assert targets.approval_mask == 0.5
    assert targets.disapproval_target == (1.0,)
    assert targets.disapproval_mask == 1.0


def test_weak_row_to_targets_maps_no_crowd_to_negative_event() -> None:
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 0,
                "unclear_disapproval": 0,
                "unclear_approval": 0,
                "clear_approval": 0,
                "hard_annotation": 1,
                "no_crowd": 1,
                "crowd_chorus": 0,
            }
        )
    )
    assert targets.event_target == (0.0,)
    assert targets.event_mask == 1.0
    assert targets.approval_target == (0.0,)
    assert targets.approval_mask == 1.0
    assert targets.disapproval_target == (0.0,)
    assert targets.disapproval_mask == 1.0


def test_parse_strong_label_file_ignores_non_target_labels(tmp_path: Path) -> None:
    txt_path = tmp_path / "noise_sample.txt"
    txt_path.write_text("0.0\t1.0\tclear_disapproval\n1.0\t2.0\tcrowd_chorus\n2.0\t3.0\tunclear_approval\n", encoding="utf-8")
    events = parse_strong_label_file(str(txt_path), speech_id="speech-1")
    assert [(event.event_class, event.onset_sec, event.offset_sec) for event in events] == [
        (0, 0.0, 1.0),
        (0, 1.0, 2.0),
        (0, 2.0, 3.0),
    ]


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


def test_build_audio_augmentation_disabled_returns_none() -> None:
    assert build_audio_augmentation(None) is None
    assert build_audio_augmentation({"enabled": False}) is None
    assert build_audio_augmentation({"enabled": True}) is None


def test_audiomentations_chain_preserves_shape_dtype_and_finite_values() -> None:
    pytest.importorskip("audiomentations")
    augmentation = build_audio_augmentation(
        {
            "enabled": True,
            "lowpass": {"p": 1.0, "cutoff_min_hz": 3500, "cutoff_max_hz": 3500},
            "pink_noise": {"p": 1.0, "snr_min_db": 20.0, "snr_max_db": 20.0},
            "clipping": {"p": 1.0, "min_percentile_threshold": 10, "max_percentile_threshold": 10},
        }
    )
    waveform = torch.linspace(-0.5, 0.5, 16000, dtype=torch.float32)

    augmented = apply_audio_augmentation(waveform, augmentation, sample_rate=16000)

    assert augmented.shape == waveform.shape
    assert augmented.dtype == waveform.dtype
    assert torch.isfinite(augmented).all()
    assert not torch.allclose(augmented, waveform)


def test_clipping_augmentation_changes_waveform_and_keeps_values_finite() -> None:
    pytest.importorskip("audiomentations")
    augmentation = build_audio_augmentation(
        {
            "enabled": True,
            "clipping": {"p": 1.0, "min_percentile_threshold": 40, "max_percentile_threshold": 40},
        }
    )
    waveform = torch.linspace(-1.0, 1.0, 16000, dtype=torch.float32)

    augmented = apply_audio_augmentation(waveform, augmentation, sample_rate=16000)

    assert augmented.shape == waveform.shape
    assert torch.isfinite(augmented).all()
    assert not torch.allclose(augmented, waveform)


def test_training_dataset_can_apply_augmentation_while_validation_is_unchanged(tmp_path: Path) -> None:
    pytest.importorskip("audiomentations")
    audio_path = tmp_path / "sample.wav"
    waveform = torch.linspace(-1.0, 1.0, 16000 * 20, dtype=torch.float32)
    _write_audio(audio_path, waveform)
    targets = weak_row_to_targets(
        pd.Series(
            {
                "clear_disapproval": 0,
                "unclear_disapproval": 0,
                "unclear_approval": 0,
                "clear_approval": 1,
                "no_crowd": 0,
                "crowd_chorus": 0,
            }
        )
    )
    record = WeakChunkRecord(
        audio_path=str(audio_path),
        speech_id="sample",
        chunk_start_sec=0.0,
        chunk_end_sec=20.0,
        targets=targets,
        split="train",
    )
    augmentation_config = {
        "enabled": True,
        "clipping": {"p": 1.0, "min_percentile_threshold": 40, "max_percentile_threshold": 40},
    }

    train_dataset = WeakChunkDataset([record], sample_rate=16000, chunk_sec=20.0, instance_sec=1.0, augmentation_config=augmentation_config)
    val_dataset = WeakChunkDataset([record], sample_rate=16000, chunk_sec=20.0, instance_sec=1.0)

    train_item = train_dataset[0]
    val_item = val_dataset[0]

    assert train_item["waveform"].shape == val_item["waveform"].shape
    assert not torch.allclose(train_item["waveform"], val_item["waveform"])
    assert torch.allclose(val_item["waveform"], waveform)


def test_build_split_records_uses_audio_info_for_train_val_partition(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
    )

    assert len(split_data.train_records) == 1
    assert len(split_data.val_records) == 1
    assert split_data.train_records[0].split == "train"
    assert split_data.val_records[0].split == "val"
    assert split_data.val_records[0].speech_id == "Birth control question booed at CNN Arizona debate"
    assert split_data.train_records[0].speech_id == "Taylor Swift booed at Super Bowl #taylorswift #superbowl ｜ Sports Illustrated"


def test_build_split_records_adds_strong_only_files_as_overlapping_val_chunks(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    strong_only_audio = audio_dir / "Strong only crowd reaction.wav"
    _write_audio(strong_only_audio, torch.ones(16000 * 36))
    (strong_dir / "noise_Strong only crowd reaction.txt").write_text(
        "2.0\t4.0\tclear_approval\n22.0\t24.0\tunclear_disapproval\n",
        encoding="utf-8",
    )

    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
        chunk_sec=20.0,
    )

    strong_only_records = [
        record
        for record in split_data.val_records
        if record.speech_id == "Strong only crowd reaction"
    ]
    assert [(record.chunk_start_sec, record.chunk_end_sec) for record in strong_only_records] == [
        (0.0, 20.0),
        (10.0, 30.0),
        (20.0, 36.0),
    ]
    assert all(record.audio_path == str(strong_only_audio.resolve()) for record in strong_only_records)
    assert all(record.split == "val" for record in strong_only_records)
    assert all(record.targets.event_mask == 0.0 for record in strong_only_records)
    assert all(record.targets.approval_mask == 0.0 for record in strong_only_records)
    assert all(record.targets.disapproval_mask == 0.0 for record in strong_only_records)
    assert any(event.speech_id == "Strong only crowd reaction" for event in split_data.strong_events_by_task["approval"])
    assert any(event.speech_id == "Strong only crowd reaction" for event in split_data.strong_events_by_task["disapproval"])


def test_build_strong_validation_split_does_not_need_weak_labels(tmp_path: Path) -> None:
    _, _, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    strong_only_audio = audio_dir / "Strong only crowd reaction.wav"
    _write_audio(strong_only_audio, torch.ones(16000 * 25))
    (strong_dir / "noise_Strong only crowd reaction.txt").write_text(
        "2.0\t4.0\tclear_approval\n",
        encoding="utf-8",
    )

    split_data = build_strong_validation_split(
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
        chunk_sec=20.0,
    )

    assert not split_data.train_records
    records_by_speech = {}
    for record in split_data.val_records:
        records_by_speech.setdefault(record.speech_id, []).append(record)
    assert set(records_by_speech) == {
        "Birth control question booed at CNN Arizona debate",
        "Strong only crowd reaction",
    }
    assert [(record.chunk_start_sec, record.chunk_end_sec) for record in records_by_speech["Strong only crowd reaction"]] == [
        (0.0, 20.0),
        (10.0, 25.0),
    ]
    assert any(event.speech_id == "Strong only crowd reaction" for event in split_data.strong_events_by_task["approval"])


def test_build_strong_validation_split_accepts_txt_without_noise_prefix(tmp_path: Path) -> None:
    _, _, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    strong_only_audio = audio_dir / "Plain strong label name.wav"
    _write_audio(strong_only_audio, torch.ones(16000 * 12))
    (strong_dir / "Plain strong label name.txt").write_text(
        "2.0\t4.0\tclear_approval\n",
        encoding="utf-8",
    )

    split_data = build_strong_validation_split(
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
        chunk_sec=20.0,
    )

    assert any(record.speech_id == "Plain strong label name" for record in split_data.val_records)
    assert any(event.speech_id == "Plain strong label name" for event in split_data.strong_events_by_task["approval"])


def test_build_strong_validation_split_rejects_ambiguous_txt_names(tmp_path: Path) -> None:
    _, _, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    _write_audio(audio_dir / "Ambiguous label.wav", torch.ones(16000 * 12))
    (strong_dir / "Ambiguous label.txt").write_text("2.0\t4.0\tclear_approval\n", encoding="utf-8")
    (strong_dir / "noise_Ambiguous label.txt").write_text("5.0\t6.0\tclear_disapproval\n", encoding="utf-8")

    try:
        build_strong_validation_split(
            strong_labels_dir=str(strong_dir),
            original_audio_dir=str(audio_dir),
            chunk_sec=20.0,
        )
    except ValueError as exc:
        assert "Ambiguous strong label filename" in str(exc)
    else:
        raise AssertionError("Expected ambiguous strong TXT names to raise ValueError")


def test_build_split_records_uses_strong_txt_as_validation_source_of_truth(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    (strong_dir / "noise_Taylor Swift booed at Super Bowl #taylorswift #superbowl | Sports Illustrated.txt").write_text(
        "1.0\t2.0\tunclear_approval\n",
        encoding="utf-8",
    )

    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
    )

    assert not split_data.train_records
    assert {
        record.speech_id
        for record in split_data.val_records
    } == {
        "Birth control question booed at CNN Arizona debate",
        "Taylor Swift booed at Super Bowl #taylorswift #superbowl ｜ Sports Illustrated",
    }


def test_build_split_records_loads_audio_from_original_audio_dir(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
    )
    dataset = WeakChunkDataset(split_data.val_records, sample_rate=16000, chunk_sec=20.0, instance_sec=1.0)
    item = dataset[0]

    assert Path(split_data.val_records[0].audio_path).parent == audio_dir
    assert item["waveform"].shape == (16000 * 20,)
    assert item["instances"].shape == (20, 16000)
    assert torch.equal(item["targets"]["event_target"], torch.tensor([1.0]))
    assert torch.equal(item["targets"]["approval_target"], torch.tensor([0.0]))
    assert torch.equal(item["targets"]["disapproval_target"], torch.tensor([1.0]))
    assert float(item["waveform"][-1].item()) > 0.0


def test_collate_batch_stacks_structured_targets(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
    )
    dataset = WeakChunkDataset(split_data.train_records + split_data.val_records, sample_rate=16000, chunk_sec=20.0, instance_sec=1.0)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["targets"]["event_target"].shape == (2, 1)
    assert batch["targets"]["approval_target"].shape == (2, 1)
    assert batch["targets"]["disapproval_target"].shape == (2, 1)
    assert batch["targets"]["event_mask"].shape == (2,)


def test_build_split_records_appends_segmented_negative_data_to_train(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    negative_dir = tmp_path / "data" / "negative_data"
    _write_audio(negative_dir / "plain_speech_seg001.wav", torch.full((16000 * 20,), 0.25))
    _write_audio(negative_dir / "plain_speech_seg002.wav", torch.full((16000 * 20,), 0.5))

    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
        negative_data_dir=str(negative_dir),
        chunk_sec=20.0,
    )

    negative_records = [record for record in split_data.train_records if Path(record.audio_path).parent == negative_dir]
    assert len(negative_records) == 2
    assert all(record.split == "train" for record in negative_records)
    assert all(record.chunk_start_sec == 0.0 and record.chunk_end_sec == 20.0 for record in negative_records)
    assert all(record.targets.event_target == (0.0,) for record in negative_records)
    assert all(record.targets.event_mask == 1.0 for record in negative_records)
    assert all(record.targets.approval_target == (0.0,) for record in negative_records)
    assert all(record.targets.approval_mask == 1.0 for record in negative_records)
    assert all(record.targets.disapproval_target == (0.0,) for record in negative_records)
    assert all(record.targets.disapproval_mask == 1.0 for record in negative_records)

    dataset = WeakChunkDataset(negative_records, sample_rate=16000, chunk_sec=20.0, instance_sec=1.0)
    item = dataset[0]
    assert item["audio_path"] == str((negative_dir / "plain_speech_seg001.wav").resolve())
    assert item["waveform"].shape == (16000 * 20,)


def test_missing_negative_data_dir_preserves_existing_split(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    split_data = build_split_records(
        audios_info_csv=str(info_path),
        weak_labels_csv=str(weak_csv_path),
        strong_labels_dir=str(strong_dir),
        original_audio_dir=str(audio_dir),
        negative_data_dir=str(tmp_path / "data" / "missing_negative_data"),
    )

    assert len(split_data.train_records) == 1
    assert len(split_data.val_records) == 1


def test_strong_labelled_files_missing_txt_fail_loudly(tmp_path: Path) -> None:
    info_path, weak_csv_path, strong_dir, audio_dir = _write_dataset_fixture(tmp_path)
    (strong_dir / "noise_Birth control question booed at CNN Arizona debate.txt").unlink()

    try:
        build_split_records(
            audios_info_csv=str(info_path),
            weak_labels_csv=str(weak_csv_path),
            strong_labels_dir=str(strong_dir),
            original_audio_dir=str(audio_dir),
        )
    except ValueError as exc:
        assert "matching TXT" in str(exc)
    else:
        raise AssertionError("Expected missing strong TXT to raise ValueError")
