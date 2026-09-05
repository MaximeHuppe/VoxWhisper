"""Collapse FreeSurfer/HCP wmparc labels onto a SynthSeg/OpenMind ~32-tissue set."""
from __future__ import annotations

import numpy as np

# Compact ids written into mask.nii.gz (0 = background).
DENSE_STRUCTURES: list[tuple[int, str, str]] = [
    (0, "background", "background"),
    (1, "left_cerebral_white_matter", "left cerebral white matter"),
    (2, "left_cerebral_cortex", "left cerebral cortex"),
    (3, "left_lateral_ventricle", "left lateral ventricle"),
    (4, "left_inferior_lateral_ventricle", "left inferior lateral ventricle"),
    (5, "left_cerebellum_white_matter", "left cerebellum white matter"),
    (6, "left_cerebellum_cortex", "left cerebellum cortex"),
    (7, "left_thalamus", "left thalamus"),
    (8, "left_caudate", "left caudate"),
    (9, "left_putamen", "left putamen"),
    (10, "left_pallidum", "left pallidum"),
    (11, "third_ventricle", "third ventricle"),
    (12, "fourth_ventricle", "fourth ventricle"),
    (13, "brainstem", "brainstem"),
    (14, "left_hippocampus", "left hippocampus"),
    (15, "left_amygdala", "left amygdala"),
    (16, "csf", "cerebrospinal fluid"),
    (17, "left_accumbens", "left accumbens"),
    (18, "left_ventral_dc", "left ventral diencephalon"),
    (19, "right_cerebral_white_matter", "right cerebral white matter"),
    (20, "right_cerebral_cortex", "right cerebral cortex"),
    (21, "right_lateral_ventricle", "right lateral ventricle"),
    (22, "right_inferior_lateral_ventricle", "right inferior lateral ventricle"),
    (23, "right_cerebellum_white_matter", "right cerebellum white matter"),
    (24, "right_cerebellum_cortex", "right cerebellum cortex"),
    (25, "right_thalamus", "right thalamus"),
    (26, "right_caudate", "right caudate"),
    (27, "right_putamen", "right putamen"),
    (28, "right_pallidum", "right pallidum"),
    (29, "right_hippocampus", "right hippocampus"),
    (30, "right_amygdala", "right amygdala"),
    (31, "right_accumbens", "right accumbens"),
    (32, "right_ventral_dc", "right ventral diencephalon"),
]

# FreeSurfer / wmparc source id → compact dense id.
_FS_TO_DENSE: dict[int, int] = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    7: 5,
    8: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    24: 16,
    26: 17,
    28: 18,
    41: 19,
    42: 20,
    43: 21,
    44: 22,
    46: 23,
    47: 24,
    49: 25,
    50: 26,
    51: 27,
    52: 28,
    53: 29,
    54: 30,
    58: 31,
    60: 32,
    # Corpus callosum → cerebral WM (SynthSeg has no separate CC class).
    251: 1,
    252: 1,
    253: 1,
    254: 1,
    255: 1,
    5001: 1,   # unsegmented WM (left-ish in wmparc)
    5002: 19,
}


def _lookup_table() -> np.ndarray:
    """Return a vectorised LUT covering wmparc ids (0..max)."""
    max_id = 4035
    table = np.zeros(max_id + 1, dtype=np.int16)
    for src, dst in _FS_TO_DENSE.items():
        if 0 <= src <= max_id:
            table[src] = dst
    for src in range(1000, 1036):
        table[src] = 2  # left cortex
    for src in range(2000, 2036):
        table[src] = 20  # right cortex
    for src in range(3000, 3036):
        table[src] = 1  # left gyral WM → left WM
    for src in range(4000, 4036):
        table[src] = 19  # right gyral WM → right WM
    return table


_LUT = _lookup_table()


def collapse_wmparc(label_volume: np.ndarray) -> np.ndarray:
    """Map a FreeSurfer/wmparc volume onto compact dense ids (int16)."""
    src = np.rint(label_volume).astype(np.int32)
    out = np.zeros(src.shape, dtype=np.int16)
    valid = (src >= 0) & (src < len(_LUT))
    out[valid] = _LUT[src[valid]]
    return out


def dense_structures_manifest() -> dict[str, dict[str, object]]:
    """JSON-serialisable structures table (label + prompt)."""
    return {
        name: {"label": label, "prompt": prompt}
        for label, name, prompt in DENSE_STRUCTURES
    }


def n_dense_classes() -> int:
    """Number of compact labels including background."""
    return len(DENSE_STRUCTURES)
