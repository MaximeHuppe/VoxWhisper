"""Run preprocessing for the stage named in the YAML.

Phase 1 (``model.name: VoxDense``): T1 + wmparc collapse, pretrain subjects.
Phase 2 (``model.name: VoxWhisper``): T1 + FA + named nerve masks, nerve subjects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from voxwhisper.data.nifti_io import list_subject_ids, subject_is_complete
from voxwhisper.data.preprocess.embeddings import cache_embedding
from voxwhisper.data.preprocess.masks import preprocess_masks
from voxwhisper.data.preprocess.volumes import preprocess_volumes
from voxwhisper.data.splits import create_or_load_subject_split
from voxwhisper.util.config import PRIMARY_MODALITY, SECONDARY_MODALITY, load_config, resolve_path
from voxwhisper.util.stage import cohort_name, stage_id, uses_secondary


def drop_incomplete_processed(config) -> list[str]:
    """Remove processed subject folders missing the volumes this stage needs."""
    processed_dir = resolve_path(config, "data.paths.processed")
    secondary = SECONDARY_MODALITY if uses_secondary(config) else None
    subjects = list_subject_ids(processed_dir)
    removed = []
    for sid in subjects:
        if not subject_is_complete(processed_dir, sid, PRIMARY_MODALITY, secondary):
            import shutil
            subj_path = processed_dir / sid
            if subj_path.is_dir():
                shutil.rmtree(subj_path)
            removed.append(sid)
    return removed


def _subject_ids_for_stage(config, split: dict) -> list[str] | None:
    """Return the cohort list, or None when no split exists yet."""
    cohort = cohort_name(config)
    ids = split.get(cohort) or []
    if ids:
        return ids
    if split.get("pretrain") or split.get("nerve"):
        return []
    return None


def run_preprocess(config) -> None:
    """Run preprocessing for the configured stage."""
    print(f"=== Stage: {stage_id(config)} ({cohort_name(config)} cohort) ===")
    print("=== Step 1/5: pretrain / nerve subject split ===")
    stage = create_or_load_subject_split(config)
    subject_ids = _subject_ids_for_stage(config, stage)
    if subject_ids:
        print(
            f"Processing {len(subject_ids)} {cohort_name(config)} subject(s); "
            f"other cohort held out"
        )
    elif subject_ids == []:
        print(f"No {cohort_name(config)} subjects in subject_split — nothing to process")
    else:
        print("No subject split yet — processing every raw subject")

    print("=== Step 2/5: preprocess volumes ===")
    preprocess_volumes(config, subject_ids=subject_ids)

    print("=== Step 3/5: preprocess masks ===")
    preprocess_masks(config, subject_ids=subject_ids)

    print("=== Step 4/5: drop incomplete processed subjects ===")
    removed = drop_incomplete_processed(config)
    if removed:
        print(f"Removed {len(removed)} incomplete subject(s)")
    else:
        print("No incomplete subjects")

    print("=== Step 5/5: cache prompt embeddings ===")
    prompts = config["data"]["prompts"]
    model_name = config["text_encoder"]["model_name"]
    cache_dir = resolve_path(config, "data.paths.cache")
    cache_file = cache_dir / config["text_encoder"]["cache_file"]
    cache_embedding(prompts, Path(cache_file), model_name)
    print(f"Preprocessing complete. Embeddings at {cache_file}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing for the stage named in the YAML"
    )
    parser.add_argument("--config", default="config/voxdense.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_preprocess(cfg)


if __name__ == "__main__":
    main()
