"""pretrain vs nerve subject split."""
from __future__ import annotations

from pathlib import Path

from voxwhisper.data.splits import (
    build_subject_split,
    create_or_load_subject_split,
    list_processed_subjects,
)


def _touch_nii(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_subject_split_holds_out_nerve_masks(tmp_config, tmp_path):
    raw = Path(tmp_config["data"]["paths"]["raw"])
    (raw / "pre_a").mkdir(parents=True)
    (raw / "pre_b").mkdir(parents=True)
    (raw / "nerve_a").mkdir(parents=True)
    _touch_nii(raw / "nerve_a" / "nerve_masks_1.25" / "left.nii.gz")

    split = build_subject_split(tmp_config)
    assert split["pretrain"] == ["pre_a", "pre_b"]
    assert split["nerve"] == ["nerve_a"]


def test_create_or_load_does_not_write_empty_split(tmp_config):
    split_path = Path(tmp_config["splits"]["subject_split"])
    payload = create_or_load_subject_split(tmp_config)
    assert payload == {"pretrain": [], "nerve": []}
    assert not split_path.exists()


def test_list_processed_subjects_filters_to_pretrain(tmp_config):
    processed = Path(tmp_config["data"]["paths"]["processed"])
    for sid in ("pre_a", "nerve_a"):
        _touch_nii(processed / sid / "t1.nii.gz")

    split_path = Path(tmp_config["splits"]["subject_split"])
    split_path.write_text('{"pretrain": ["pre_a"], "nerve": ["nerve_a"]}', encoding="utf-8")

    assert list_processed_subjects(tmp_config) == ["pre_a"]


def test_list_processed_subjects_filters_to_nerve(tmp_whisper_config):
    processed = Path(tmp_whisper_config["data"]["paths"]["processed"])
    for sid in ("pre_a", "nerve_a"):
        _touch_nii(processed / sid / "t1.nii.gz")

    split_path = Path(tmp_whisper_config["splits"]["subject_split"])
    split_path.write_text('{"pretrain": ["pre_a"], "nerve": ["nerve_a"]}', encoding="utf-8")

    assert list_processed_subjects(tmp_whisper_config) == ["nerve_a"]
