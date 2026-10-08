"""Public API error boundary without exposing internal exception details.

This example was independently adapted from lessons in private gateway work.
It contains no operational route, service name, credential or gateway schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The caller is allowed to see *only* these messages. Never echo an
# upstream "detail", URL, stack trace, internal ID, header or token.
_SAFE_MESSAGES: dict[str, str] = {
    "BAD_INPUT": "The request could not be accepted.",
    "NOT_AUTHORIZED": "The requested operation is not authorized.",
    "STATE_CONFLICT": "The request conflicts with the current state.",
    "SERVICE_UNAVAILABLE": "The service is temporarily unavailable.",
}


@dataclass(frozen=True, slots=True)
class PublicError:
    code: str
    message: str
    http_status: int
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        """Explicit allowlist, rather than serializing upstream objects."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
            "status": self.http_status,
        }


def parse_known_error(status: object, payload: object) -> PublicError | None:
    """Parse a narrowly recognized error; refuse unknown shapes.

    Supported inputs: {"error": "CODE"} or {"error": {"code": "CODE"}}.
    Returning None means "unrecognized", not "success".

    This is a transport demonstration, not authentication or error recovery.
    """
    if type(status) is not int or not 400 <= status <= 599:
        return None
    if not isinstance(payload, Mapping):
        return None

    raw_error = payload.get("error")
    if isinstance(raw_error, str):
        code = raw_error
        fields = payload
    elif isinstance(raw_error, Mapping):
        code = raw_error.get("code")
        fields = raw_error
    else:
        return None

    if not isinstance(code, str) or code not in _SAFE_MESSAGES:
        return None

    # External truthy strings such as "true" cannot force retry behavior.
    # Retrying is possible only on a known code and known transient statuses.
    retryable = (
        code == "SERVICE_UNAVAILABLE"
        and status in (502, 503, 504)
        and fields.get("retryable") is True
    )

    return PublicError(
        code=code,
        message=_SAFE_MESSAGES[code],
        http_status=status,
        retryable=retryable,
    )