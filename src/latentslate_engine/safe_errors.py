"""Closed, safe-to-publish generation failures.

These errors carry only Engine-owned labels.  They deliberately form a narrow
exception seam so a worker diagnostic can be returned to the API without
making the normal unexpected-exception logging path less useful.
"""

from __future__ import annotations


class SafeJobFailure(RuntimeError):
    """A generation failure whose message and log labels are already sanitized."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        error_type: str,
        diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.error_type = error_type
        self.diagnostic = diagnostic
