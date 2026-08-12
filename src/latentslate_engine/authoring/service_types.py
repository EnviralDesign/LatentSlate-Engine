from __future__ import annotations

import re
from typing import Literal

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_INSTALL_STATE_TOKENS = (
    "not installed",
    "is incomplete",
    "missing resource",
    "required artifact",
    "does not exist",
    "no compatible model resources were discovered",
)
ActivationAction = Literal["next_cli_invocation", "restart_engine"]


class CatalogAuthoringError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_authoring_request") -> None:
        super().__init__(message)
        self.code = code
