"""Create leakage-safe CT image manifests by splitting complete groups."""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(".")
DATASET_ROOT = Path(
    "data/raw/brain_ct/Brain_Data_Organised"
)
OUTPUT_DIR = Path("data/processed")

CLASS_LABELS = {"Normal": 0, "Stroke": 1}
SPLIT_NAMES = ("train", "validation", "test")
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EXPECTED_IMAGES = 2501
EXPECTED_CLASS_IMAGES = {"Normal": 1551, "Stroke": 950}
EXPECTED_GROUPS = 82
EXPECTED_CLASS_GROUPS = {"Normal": 51, "Stroke": 31}

MANIFEST_COLUMNS = (
    "relative_path",
    "class_name",
    "label",
    "group_id",
    "split",
)
GROUP_PATTERN = re.compile(
    r"^(?P<group>.+?)\s*\((?P<index>\d+)\)\.jpg$",
    flags=re.IGNORECASE,
)


def extract_group_id(filename, class_name):
    """Extract a class-qualified group identifier from a CT filename."""
    if class_name not in CLASS_LABELS:
        raise ValueError(f"Unknown class: {class_name}")

    match = GROUP_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(
            "Expected filename '<group> (<index>).jpg', "
            f"received: {filename}"
        )

    group = match.group("group").strip()
    if not group:
        raise ValueError(f"Empty group in filename: {filename}")
    return f"{class_name}_{group}"


def _slice_index(filename):
    match = GROUP_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Invalid CT filename: {filename}")
    return int(match.group("index"))


def inventory_images(dataset_root=DATASET_ROOT, project_root=PROJECT_ROOT):
    """Inventory direct JPG children of the two expected class folders."""
    dataset_root = Path(dataset_root).resolve()
    project_root = Path(project_root).resolve()
    rows = []

    for class_name, label in CLASS_LABELS.items():
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        entries = list(class_dir.iterdir())
        unexpected = [
            path
            for path in entries
            if not path.is_file() or path.suffix.lower() != ".jpg"
        ]
        if unexpected:
            raise ValueError(
                f"Unexpected entries in {class_dir}: {unexpected}"
            )

        image_paths = sorted(
            path for path in entries if path.is_file()
        )
        for image_path in image_paths:
            try:
                relative_path = image_path.resolve().relative_to(
                    project_root
                )
            except ValueError as error:
                raise ValueError(
                    f"Image is outside project root: {image_path}"
                ) from error

            rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "class_name": class_name,
                    "label": label,
                    "group_id": extract_group_id(
                        image_path.name,
                        class_name,
                    ),
                    "slice_index": _slice_index(image_path.name),
                }
            )

    inventory = pd.DataFrame(rows)
    return inventory.sort_values(
        by=["label", "group_id", "slice_index", "relative_path"],
        kind="stable",
    ).reset_index(drop=True)


def validate_real_inventory(inventory):
    """Validate the audited dataset counts before producing manifests."""
    if len(inventory) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, found {len(inventory)}."
        )

    class_counts = inventory["class_name"].value_counts().to_dict()
    if class_counts != EXPECTED_CLASS_IMAGES:
        raise ValueError(
            "Unexpected image counts by class: "
            f"{class_counts}"
        )

    groups = inventory.drop_duplicates("group_id")
    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_GROUPS} groups, found {len(groups)}."
        )

    group_counts = groups["class_name"].value_counts().to_dict()
    if group_counts != EXPECTED_CLASS_GROUPS:
        raise ValueError(
            "Unexpected group counts by class: "
            f"{group_counts}"
        )


def split_inventory(inventory, seed=RANDOM_SEED):
    """Assign complete, class-stratified groups to each split."""
    classes_per_group = inventory.groupby("group_id")[
        "class_name"
    ].nunique()
    if (classes_per_group != 1).any():
        raise ValueError("A group_id is associated with multiple classes.")

    groups = (
        inventory[["group_id", "class_name", "label"]]
        .drop_duplicates()
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )
    development, test = train_test_split(
        groups,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=groups["label"],
    )
    train, validation = train_test_split(
        development,
        test_size=VALIDATION_SIZE,
        random_state=seed,
        stratify=development["label"],
    )

    assignments = {}
    for split_name, split_groups in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        assignments.update(
            dict.fromkeys(split_groups["group_id"], split_name)
        )

    manifest = inventory.copy()
    manifest["split"] = manifest["group_id"].map(assignments)
    manifests = {}
    for split_name in SPLIT_NAMES:
        manifests[split_name] = (
            manifest[manifest["split"] == split_name]
            .sort_values(
                by=["label", "group_id", "slice_index", "relative_path"],
                kind="stable",
            )
            .loc[:, MANIFEST_COLUMNS]
            .reset_index(drop=True)
        )
    return manifests


def validate_manifests(manifests, expected_total=None):
    """Validate schema, labels and absolute group/image separation."""
    if set(manifests) != set(SPLIT_NAMES):
        raise ValueError(f"Expected manifests for {SPLIT_NAMES}.")

    combined = []
    groups_by_split = {}
    images_by_split = {}
    for split_name in SPLIT_NAMES:
        manifest = manifests[split_name]
        if tuple(manifest.columns) != MANIFEST_COLUMNS:
            raise ValueError(f"Invalid schema for {split_name} manifest.")
        if set(manifest["split"]) != {split_name}:
            raise ValueError(f"Invalid split values in {split_name}.")
        if not set(manifest["class_name"]).issubset(CLASS_LABELS):
            raise ValueError(f"Unknown class in {split_name}.")
        expected_labels = manifest["class_name"].map(CLASS_LABELS)
        if not manifest["label"].equals(expected_labels):
            raise ValueError(f"Inconsistent labels in {split_name}.")
        if manifest["relative_path"].duplicated().any():
            raise ValueError(f"Duplicate image in {split_name}.")

        groups_by_split[split_name] = set(manifest["group_id"])
        images_by_split[split_name] = set(manifest["relative_path"])
        combined.append(manifest)

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if groups_by_split[left] & groups_by_split[right]:
                raise ValueError(f"Group leakage between {left} and {right}.")
            if images_by_split[left] & images_by_split[right]:
                raise ValueError(f"Image overlap between {left} and {right}.")

    all_rows = pd.concat(combined, ignore_index=True)
    if all_rows["relative_path"].duplicated().any():
        raise ValueError("An image appears in multiple manifests.")
    if expected_total is not None and len(all_rows) != expected_total:
        raise ValueError(
            f"Expected {expected_total} manifest rows, found {len(all_rows)}."
        )


def save_manifests(manifests, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        manifests[split_name].to_csv(
            output_dir / f"brain_ct_{split_name}.csv",
            index=False,
        )


def build_manifests(
    dataset_root=DATASET_ROOT,
    project_root=PROJECT_ROOT,
    seed=RANDOM_SEED,
    validate_expected=True,
):
    inventory = inventory_images(dataset_root, project_root)
    if validate_expected:
        validate_real_inventory(inventory)
    manifests = split_inventory(inventory, seed=seed)
    validate_manifests(
        manifests,
        expected_total=EXPECTED_IMAGES if validate_expected else len(inventory),
    )
    return manifests


def main():
    manifests = build_manifests()
    save_manifests(manifests)

    print("=== BRAIN CT GROUP SPLIT COMPLETED ===")
    for split_name in SPLIT_NAMES:
        manifest = manifests[split_name]
        print(
            f"{split_name}: {len(manifest)} images | "
            f"{manifest['group_id'].nunique()} groups | "
            f"Normal={(manifest['label'] == 0).sum()} | "
            f"Stroke={(manifest['label'] == 1).sum()}"
        )


if __name__ == "__main__":
    main()
