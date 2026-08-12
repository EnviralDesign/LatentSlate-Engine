"""Static, same-origin delivery for the resource authoring page.

The page is intentionally public so a local browser can load its shell without
requiring a bearer token. It only calls same-origin ``/v1`` endpoints, which
continue to use the application's existing bearer authentication dependency.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

AUTHORING_WEB_PATH = "/authoring/"
_ASSET_ROOT = Path(__file__).resolve().parent / "authoring_web"
_ASSETS = {
    "app.js": "application/javascript; charset=utf-8",
    "favicon.svg": "image/svg+xml",
    "styles.css": "text/css; charset=utf-8",
}
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'; object-src 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def register_authoring_web_routes(app: FastAPI) -> None:
    """Register the deliberately small, non-browsable static asset surface."""

    @app.get(AUTHORING_WEB_PATH, include_in_schema=False)
    async def authoring_page() -> FileResponse:
        return _asset_response("index.html", cache_control="no-store")

    @app.get("/authoring/{asset_name}", include_in_schema=False)
    async def authoring_asset(asset_name: str) -> FileResponse:
        media_type = _ASSETS.get(asset_name)
        if media_type is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _asset_response(
            asset_name,
            media_type=media_type,
            cache_control="no-store",
        )


def _asset_response(
    asset_name: str,
    *,
    media_type: str = "text/html; charset=utf-8",
    cache_control: str,
) -> FileResponse:
    """Return one known file without exposing a directory or accepting paths."""

    path = _ASSET_ROOT / asset_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Authoring page is unavailable")
    return FileResponse(
        path,
        media_type=media_type,
        headers={**_SECURITY_HEADERS, "Cache-Control": cache_control},
    )
