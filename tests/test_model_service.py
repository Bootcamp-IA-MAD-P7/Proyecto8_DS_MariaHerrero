import json

import joblib
import pytest

from src.api.services.model_service import ModelService


def create_model_file(path):
    model = {"name": "temporary-model"}
    joblib.dump(model, path)

    return model


def test_load_reads_temporary_model_and_metadata(tmp_path):
    model_path = tmp_path / "model.joblib"
    threshold_path = tmp_path / "threshold.json"
    expected_model = create_model_file(model_path)
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": 0.37,
                "model_version": "test_v1",
            }
        ),
        encoding="utf-8",
    )
    service = ModelService(
        model_path=model_path,
        threshold_path=threshold_path,
    )

    service.load()

    assert service.is_loaded is True
    assert service.get_model() == expected_model
    assert service.threshold == 0.37
    assert service.model_version == "test_v1"


def test_load_raises_when_model_does_not_exist(tmp_path):
    service = ModelService(
        model_path=tmp_path / "missing.joblib",
        threshold_path=tmp_path / "threshold.json",
    )

    with pytest.raises(
        FileNotFoundError,
        match="No se encontró el modelo final",
    ):
        service.load()


def test_load_raises_when_threshold_metadata_does_not_exist(
    tmp_path,
):
    model_path = tmp_path / "model.joblib"
    create_model_file(model_path)
    service = ModelService(
        model_path=model_path,
        threshold_path=tmp_path / "missing.json",
    )

    with pytest.raises(
        FileNotFoundError,
        match="No se encontró la configuración del threshold",
    ):
        service.load()


def test_load_raises_for_invalid_metadata_json(tmp_path):
    model_path = tmp_path / "model.joblib"
    threshold_path = tmp_path / "threshold.json"
    create_model_file(model_path)
    threshold_path.write_text(
        "not valid json",
        encoding="utf-8",
    )
    service = ModelService(
        model_path=model_path,
        threshold_path=threshold_path,
    )

    with pytest.raises(json.JSONDecodeError):
        service.load()


def test_load_raises_when_threshold_key_is_missing(tmp_path):
    model_path = tmp_path / "model.joblib"
    threshold_path = tmp_path / "threshold.json"
    create_model_file(model_path)
    threshold_path.write_text(
        json.dumps({"model_version": "test_v1"}),
        encoding="utf-8",
    )
    service = ModelService(
        model_path=model_path,
        threshold_path=threshold_path,
    )

    with pytest.raises(KeyError, match="threshold"):
        service.load()


def test_get_model_raises_before_load(tmp_path):
    service = ModelService(
        model_path=tmp_path / "model.joblib",
        threshold_path=tmp_path / "threshold.json",
    )

    with pytest.raises(
        RuntimeError,
        match="El modelo no está cargado",
    ):
        service.get_model()
