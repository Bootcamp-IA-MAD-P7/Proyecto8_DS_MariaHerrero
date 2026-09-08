from pathlib import Path

import pandas as pd
import pytest

from src.data.split_brain_ct import (
    CLASS_LABELS,
    MANIFEST_COLUMNS,
    SPLIT_NAMES,
    build_manifests,
    extract_group_id,
    inventory_images,
    split_inventory,
    validate_manifests,
)


def create_synthetic_dataset(tmp_path, groups_per_class=20):
    dataset_root = tmp_path / "brain_ct"
    for class_name in CLASS_LABELS:
        class_dir = dataset_root / class_name
        class_dir.mkdir(parents=True)
        for group in range(groups_per_class):
            for index in range(1, 4):
                (class_dir / f"{group} ({index}).jpg").write_bytes(b"jpg")
    return dataset_root


def test_extract_group_id_qualifies_group_with_class():
    assert extract_group_id("100 (14).jpg", "Normal") == "Normal_100"
    assert extract_group_id("58 (1).jpg", "Stroke") == "Stroke_58"


def test_inventory_maps_classes_and_uses_relative_paths(tmp_path):
    dataset_root = create_synthetic_dataset(tmp_path, groups_per_class=2)
    inventory = inventory_images(dataset_root, project_root=tmp_path)

    assert set(inventory.groupby("class_name")["label"].first().items()) == {
        ("Normal", 0),
        ("Stroke", 1),
    }
    assert all(
        not Path(path).is_absolute()
        for path in inventory["relative_path"]
    )


def test_group_split_is_reproducible(tmp_path):
    dataset_root = create_synthetic_dataset(tmp_path)
    inventory = inventory_images(dataset_root, project_root=tmp_path)

    first = split_inventory(inventory, seed=42)
    second = split_inventory(inventory, seed=42)

    for split_name in SPLIT_NAMES:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])


def test_split_has_no_group_leakage_or_image_overlap(tmp_path):
    dataset_root = create_synthetic_dataset(tmp_path)
    manifests = build_manifests(
        dataset_root,
        project_root=tmp_path,
        validate_expected=False,
    )

    validate_manifests(manifests, expected_total=120)
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            assert set(manifests[left]["group_id"]).isdisjoint(
                manifests[right]["group_id"]
            )
            assert set(manifests[left]["relative_path"]).isdisjoint(
                manifests[right]["relative_path"]
            )


def test_each_split_has_both_classes_and_reasonable_proportions(tmp_path):
    dataset_root = create_synthetic_dataset(tmp_path)
    manifests = build_manifests(
        dataset_root,
        project_root=tmp_path,
        validate_expected=False,
    )
    expected = {"train": 0.64, "validation": 0.16, "test": 0.20}

    total = sum(len(manifest) for manifest in manifests.values())
    for split_name, manifest in manifests.items():
        assert set(manifest["class_name"]) == set(CLASS_LABELS)
        assert abs(len(manifest) / total - expected[split_name]) <= 0.05


def test_manifest_schema_and_global_counts(tmp_path):
    dataset_root = create_synthetic_dataset(tmp_path)
    manifests = build_manifests(
        dataset_root,
        project_root=tmp_path,
        validate_expected=False,
    )

    assert sum(len(manifest) for manifest in manifests.values()) == 120
    for split_name, manifest in manifests.items():
        assert tuple(manifest.columns) == MANIFEST_COLUMNS
        assert set(manifest["split"]) == {split_name}


def test_invalid_filename_fails_explicitly():
    with pytest.raises(ValueError, match="Expected filename"):
        extract_group_id("image.jpg", "Normal")
