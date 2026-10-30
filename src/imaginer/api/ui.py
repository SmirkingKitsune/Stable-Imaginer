"""The browser UI: static files under ``/ui``, its page at ``/``.

The UI is plain ES modules with no build step, and it talks to ``/v1`` like any other
client, so the bearer token (when configured) is entered in the page, not checked here.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_NO_CACHE = {"Cache-Control": "no-cache"}


class _UiFiles(StaticFiles):
    """Revalidated on every load (cheap 304s), so an upgrade never leaves stale scripts."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(_NO_CACHE)
        return response


def install_ui(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers=_NO_CACHE)

    app.mount("/ui", _UiFiles(directory=WEB_DIR), name="ui")
