import hashlib
import json
from zipfile import ZipFile

import pytest

from airfoil_surrogate.data.stage_original import stage_partition, zip_index_sha256


@pytest.fixture
def pinned_archive(tmp_path):
    case_ids = [f"airFoil2D_SST_60_0_1_4_{i}" for i in range(1000)]
    selected = [case_ids[i] for i in (0, 1, 720, 800)]
    split_path = "Dataset/manifest.json"
    splits = json.dumps(
        {"full_train": case_ids[:800], "full_test": case_ids[800:]}
    ).encode()
    path = tmp_path / "Dataset.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(split_path, splits)
        for case_id in selected:
            for suffix in ("internal.vtu", "aerofoil.vtp"):
                name = f"{case_id}_{suffix}"
                archive.writestr(f"Dataset/{case_id}/{name}", name.encode())
    with ZipFile(path) as archive:
        pin = {
            "zip_index_sha256": zip_index_sha256(archive),
            "split_path": split_path,
            "split_sha256": hashlib.sha256(splits).hexdigest(),
        }
        yield archive, pin, selected


def test_training_stage_preserves_order_and_opens_only_selected_members(
    pinned_archive, tmp_path, monkeypatch
):
    archive, pin, (first, second, _, _) = pinned_archive
    expected_members = [pin["split_path"]] + [
        f"Dataset/{case_id}/{case_id}_{suffix}"
        for case_id in (first, second)
        for suffix in ("internal.vtu", "aerofoil.vtp")
    ]
    opened = []
    original_open = archive.open

    def log_open(member, *args, **kwargs):
        opened.append(member if isinstance(member, str) else member.filename)
        return original_open(member, *args, **kwargs)

    monkeypatch.setattr(archive, "open", log_open)
    out = tmp_path / "train"
    manifest = stage_partition(archive, pin, "train", out, case_ids=[second, first])
    staged = json.loads(manifest.read_text())
    assert opened == expected_members
    assert staged["partition"] == "train"
    assert staged["partition_size"] == 720
    assert [row["id"] for row in staged["cases"]] == [first, second]
    for row in staged["cases"]:
        case_id = row["id"]
        assert set(row["files"]) == {"internal.vtu", "aerofoil.vtp"}
        for suffix, integrity in row["files"].items():
            name = f"{case_id}_{suffix}"
            expected = name.encode()
            assert (out / case_id / name).read_bytes() == expected
            assert integrity == {
                "size": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
            }
    with pytest.raises(FileExistsError):
        stage_partition(archive, pin, "train", out, case_ids=[first])
    assert json.loads(manifest.read_text()) == staged


def test_staging_rejects_wrong_partition_and_checksum_before_writing(
    pinned_archive, tmp_path
):
    archive, pin, (first, _, validation, test) = pinned_archive
    out = tmp_path / "stage"
    for wrong in (validation, test):
        with pytest.raises(ValueError, match="train partition"):
            stage_partition(archive, pin, "train", out, case_ids=[wrong])
    wrong_pin = pin | {"split_sha256": "0" * 64}
    with pytest.raises(ValueError, match="split checksum mismatch"):
        stage_partition(archive, wrong_pin, "train", out, case_ids=[first])
    assert not out.exists()
