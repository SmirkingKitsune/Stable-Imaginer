"""Typed application errors.

Each error carries a stable machine-readable ``code`` and the HTTP status the API layer
should answer with. The API renders them as RFC 9457 ``application/problem+json``.
"""

from typing import Any, ClassVar


class ImaginerError(Exception):
    status: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"
    title: ClassVar[str] = "Internal error"

    def __init__(self, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra


class NotFoundError(ImaginerError):
    status = 404
    code = "not_found"
    title = "Not found"


class ConflictError(ImaginerError):
    status = 409
    code = "conflict"
    title = "Conflict"


class JobCancelledError(ImaginerError):
    """A job was cancelled on its own (not as part of cancelling its turn)."""

    status = 409
    code = "job_cancelled"
    title = "Job cancelled"


class InvalidRequestError(ImaginerError):
    status = 422
    code = "invalid_request"
    title = "Invalid request"


class CapabilityUnavailableError(ImaginerError):
    """No configured backend offers the capability a request needs."""

    status = 422
    code = "capability_unavailable"
    title = "Capability unavailable"


class PayloadTooLargeError(ImaginerError):
    status = 413
    code = "payload_too_large"
    title = "Payload too large"


class UnsupportedMediaTypeError(ImaginerError):
    status = 415
    code = "unsupported_media_type"
    title = "Unsupported media type"


class UnauthorizedError(ImaginerError):
    status = 401
    code = "unauthorized"
    title = "Unauthorized"


class ForbiddenError(ImaginerError):
    """Authenticated, but not allowed: e.g. a linked instance on a route only the user may use."""

    status = 403
    code = "forbidden"
    title = "Forbidden"


class PairingError(ImaginerError):
    """Pairing with another instance failed: wrong or expired code, unknown challenge, ..."""

    status = 422
    code = "pairing_failed"
    title = "Pairing failed"


class TooManyRequestsError(ImaginerError):
    status = 429
    code = "too_many_requests"
    title = "Too many requests"


class EngineUnavailableError(ImaginerError):
    """The engine process could not be reached (refused, DNS, reset, still loading)."""

    status = 503
    code = "engine_unavailable"
    title = "Engine unavailable"


class EngineTimeoutError(ImaginerError):
    status = 504
    code = "engine_timeout"
    title = "Engine timed out"


class EngineError(ImaginerError):
    """The engine answered, but with an error."""

    status = 502
    code = "engine_error"
    title = "Engine error"


class EngineInterruptedError(EngineError):
    """The engine's stream ended before it said it was finished."""

    code = "engine_interrupted"
    title = "Engine stream interrupted"
