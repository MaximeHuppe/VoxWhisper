"""FreeSurfer wmparc collapse and prompt sampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voxwhisper.data.dataset import sample_prompt_labels
from voxwhisper.data.preprocess.freesurfer import (
    DENSE_STRUCTURES,
    collapse_wmparc,
    n_dense_classes,
)


def test_collapse_maps_aseg_and_gyral_parcels():
    src = np.array(
        [
            [0, 2, 3, 17],          # bg, left WM, left cortex, left hippocampus
            [41, 1003, 3001, 16],   # right WM, left gyrus GM, left gyral WM, brainstem
            [2005, 4035, 251, 999], # right gyrus GM, right gyral WM, CC, unknown
        ],
        dtype=np.int32,
    )
    out = collapse_wmparc(src)
    assert out[0, 0] == 0
    assert out[0, 1] == 1    # left WM
    assert out[0, 2] == 2    # left cortex
    assert out[0, 3] == 14   # left hippocampus
    assert out[1, 0] == 19   # right WM
    assert out[1, 1] == 2    # aparc 1003 → left cortex
    assert out[1, 2] == 1    # wmparc 3001 → left WM
    assert out[1, 3] == 13   # brainstem
    assert out[2, 0] == 20   # right cortex
    assert out[2, 1] == 19   # right gyral WM
    assert out[2, 2] == 1    # CC → left WM (SynthSeg has no CC)
    assert out[2, 3] == 0    # unknown id stays background


def test_dense_ontology_has_32_foreground_tissues():
    assert n_dense_classes() == 33
    labels = [lab for lab, _, _ in DENSE_STRUCTURES]
    assert labels == list(range(33))


def test_sample_prompt_labels_prefers_present():
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[2:5, 2:5, 2:5] = 7
    rng = np.random.default_rng(0)
    chosen = sample_prompt_labels(labels, positive_labels=[7, 8, 9], k=1, rng=rng)
    assert chosen == [7]

    chosen_k = sample_prompt_labels(labels, positive_labels=[7, 8, 9], k=3, rng=rng)
    assert len(chosen_k) == 3
    assert 7 in chosen_k


def test_process_named_masks_stacks_on_t1(tmp_whisper_config, tmp_path):
    import nibabel as nib
    from voxwhisper.data.nifti_io import mask_path
    from voxwhisper.data.preprocess.masks import process_mask
    from voxwhisper.util.config import load_structures

    raw = Path(tmp_whisper_config["data"]["paths"]["raw"])
    processed = Path(tmp_whisper_config["data"]["paths"]["processed"])
    sid = "nerve_a"
    t1_dir = raw / sid
    t1_dir.mkdir(parents=True)
    affine = np.eye(4)
    t1 = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(t1, affine), str(t1_dir / "T1w.nii.gz"))

    nerve = np.zeros((8, 8, 8), dtype=np.uint8)
    nerve[2:5, 2:5, 2:5] = 1
    mask_dir = t1_dir / "nerve_masks_1.25"
    mask_dir.mkdir()
    nib.save(nib.Nifti1Image(nerve, affine), str(mask_dir / "left_nerve.nii.gz"))

    structures_path = tmp_path / "nerves.json"
    structures_path.write_text(
        '{"background": {"label": 0, "prompt": "background"},'
        ' "left_nerve": {"label": 1, "prompt": "left nerve"},'
        ' "right_nerve": {"label": 2, "prompt": "right nerve"}}',
        encoding="utf-8",
    )
    tmp_whisper_config["data"]["masks"]["structures"] = str(structures_path)
    assert load_structures(tmp_whisper_config)["positive_labels"] == [1, 2]

    process_mask(sid, str(raw), processed, tmp_whisper_config)
    out = mask_path(processed, sid)
    assert out.exists()
    labels = np.rint(nib.load(str(out)).get_fdata()).astype(np.int16)
    assert int(labels[3, 3, 3]) == 1
    assert int(labels[0, 0, 0]) == 0
