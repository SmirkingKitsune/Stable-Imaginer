"""RFC 9457 problem details for every error the API returns."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from imaginer.core.logging import get_logger
from imaginer.domain.errors import ImaginerError, UnauthorizedError

log = get_logger(__name__)

PROBLEM_JSON = "application/problem+json"


def problem_body(
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    request: Request,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": f"urn:imaginer:error:{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "code": code,
        "request_id": getattr(request.state, "request_id", None),
    }
    # Extension members add context; they never replace the standard members.
    for key, value in jsonable_encoder(extra or {}).items():
        body.setdefault(key, value)
    return body


def _problem(
    status: int,
    code: str,
    title: str,
    detail: str,
    request: Request,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body = problem_body(
        status=status, code=code, title=title, detail=detail, request=request, extra=extra
    )
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


async def _imaginer_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ImaginerError)
    headers = {"WWW-Authenticate": "Bearer"} if isinstance(exc, UnauthorizedError) else None
    if exc.status >= 500:
        log.warning("request.engine_error", code=exc.code, detail=exc.detail)
    return _problem(exc.status, exc.code, exc.title, exc.detail, request, headers, exc.extra)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return _problem(
        422,
        "validation_error",
        "Request validation failed",
        "the request did not match the expected schema",
        request,
        extra={"errors": exc.errors()},
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return _problem(
        exc.status_code,
        "http_error",
        str(exc.detail),
        str(exc.detail),
        request,
        headers=dict(exc.headers) if exc.headers else None,
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("request.unhandled_error", exc_info=exc)
    return _problem(
        500, "internal_error", "Internal error", "an unexpected error occurred", request
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ImaginerError, _imaginer_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
