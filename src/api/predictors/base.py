"""Minimal structural contract shared by prediction implementations."""

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    """Describe an inference component without coupling to its modality."""

    @property
    def modality(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def model_version(self) -> str | None:
        ...

    @property
    def is_ready(self) -> bool:
        ...

    def predict(self, input_data: Any) -> Mapping[str, object]:
        ...
