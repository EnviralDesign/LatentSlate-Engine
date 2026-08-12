from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from latentslate_engine import __main__ as engine_cli
from latentslate_engine import authoring_web_routes


@pytest.fixture
def web_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assets = tmp_path / "authoring_web"
    assets.mkdir()
    (assets / "index.html").write_text("<!doctype html><title>Resource Editor</title>")
    (assets / "app.js").write_text("console.log('editor')")
    (assets / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (assets / "styles.css").write_text("body { color: black; }")
    monkeypatch.setattr(authoring_web_routes, "_ASSET_ROOT", assets)
    return assets


def test_authoring_page_is_public_and_has_safe_response_headers(web_assets: Path) -> None:
    app = FastAPI()
    authoring_web_routes.register_authoring_web_routes(app)
    with TestClient(app) as client:
        response = client.get("/authoring/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/authoring/app.js", "application/javascript"),
        ("/authoring/favicon.svg", "image/svg+xml"),
        ("/authoring/styles.css", "text/css"),
    ],
)
def test_authoring_assets_have_explicit_type_and_cache_headers(
    web_assets: Path, path: str, content_type: str
) -> None:
    app = FastAPI()
    authoring_web_routes.register_authoring_web_routes(app)
    with TestClient(app) as client:
        response = client.get(path, follow_redirects=False)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "path",
    [
        "/authoring",
        "/authoring/unknown.txt",
        "/authoring/../pyproject.toml",
        "/authoring/%2e%2e/pyproject.toml",
    ],
)
def test_authoring_routes_never_expose_directories_or_arbitrary_files(
    web_assets: Path, path: str
) -> None:
    app = FastAPI()
    authoring_web_routes.register_authoring_web_routes(app)
    with TestClient(app) as client:
        response = client.get(path, follow_redirects=False)

    if path == "/authoring":
        assert response.status_code == 307
        assert response.headers["location"].endswith("/authoring/")
    else:
        assert response.status_code == 404


def test_author_command_checks_existing_server_then_opens_default_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    requests: list[str] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def open(self, request, *, timeout: int):
            requests.append(request.full_url)
            assert timeout == 2
            return Response()

    def fake_build_opener(*_handlers):
        return Opener()

    monkeypatch.setattr(engine_cli, "build_opener", fake_build_opener)
    monkeypatch.setattr(engine_cli.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(sys, "argv", ["latentslate-engine", "author"])

    engine_cli.main()

    assert requests == ["http://127.0.0.1:8765/authoring/"]
    assert opened == requests


@pytest.mark.parametrize(
    ("raw_url", "expected_url"),
    [
        ("http://localhost:9000", "http://localhost:9000/authoring/"),
        ("https://127.0.0.1:9000/authoring", "https://127.0.0.1:9000/authoring/"),
        ("http://[::1]:9000/authoring/", "http://[::1]:9000/authoring/"),
    ],
)
def test_author_url_allows_only_normalized_loopback_pages(raw_url: str, expected_url: str) -> None:
    assert engine_cli.normalize_authoring_url(raw_url) == expected_url


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://example.com/authoring/",
        "http://192.168.1.10/authoring/",
        "file:///authoring/",
        "http://localhost:9000/other",
        "http://localhost:9000/authoring/?token=secret",
        "http://localhost@evil.example/authoring/",
    ],
)
def test_author_url_rejects_remote_or_non_authoring_targets(raw_url: str) -> None:
    with pytest.raises(ValueError, match="loopback HTTP"):
        engine_cli.normalize_authoring_url(raw_url)


def test_author_command_never_requests_or_opens_a_rejected_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_cli, "build_opener", lambda *_args: pytest.fail("network used")
    )
    monkeypatch.setattr(
        engine_cli.webbrowser, "open", lambda _url: pytest.fail("browser opened")
    )

    with pytest.raises(ValueError, match="loopback HTTP"):
        engine_cli.open_authoring_page("https://example.com/authoring/")


def test_author_command_refuses_a_loopback_redirect_without_following_or_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    class Opener:
        def open(self, request, *, timeout: int):
            requested.append(request.full_url)
            assert timeout == 2
            raise HTTPError(
                request.full_url,
                302,
                "Found",
                {"Location": "https://example.com/authoring/"},
                None,
            )

    monkeypatch.setattr(engine_cli, "build_opener", lambda *_handlers: Opener())
    monkeypatch.setattr(
        engine_cli.webbrowser, "open", lambda _url: pytest.fail("browser opened")
    )

    with pytest.raises(RuntimeError, match="not reachable"):
        engine_cli.open_authoring_page("http://127.0.0.1:8765/authoring/")

    assert requested == ["http://127.0.0.1:8765/authoring/"]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_no_redirect_handler_refuses_every_redirect_target(status: int) -> None:
    request = engine_cli.Request("http://127.0.0.1:8765/authoring/")

    with pytest.raises(HTTPError, match="Redirects are not allowed") as result:
        getattr(engine_cli._NoRedirectHandler(), f"http_error_{status}")(
            request,
            None,
            status,
            "Found",
            {"Location": "https://example.com/authoring/"},
        )

    assert result.value.url == request.full_url
    assert result.value.code == status


def test_author_command_uses_override_and_never_opens_when_engine_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise URLError("down")

    class Opener:
        def open(self, *_args, **_kwargs):
            return fail_urlopen()

    monkeypatch.setattr(engine_cli, "build_opener", lambda *_handlers: Opener())
    monkeypatch.setattr(engine_cli.webbrowser, "open", lambda _url: pytest.fail("browser opened"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["latentslate-engine", "author", "--url", "http://localhost:9000/authoring/"],
    )

    with pytest.raises(SystemExit) as result:
        engine_cli.main()

    assert result.value.code == 2


def test_current_vanilla_page_uses_same_origin_assets_and_no_embedded_secret() -> None:
    index = (authoring_web_routes._ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    script = (authoring_web_routes._ASSET_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'href="styles.css"' in index
    assert 'href="favicon.svg"' in index
    assert 'src="app.js"' in index
    assert "http://" not in script
    assert "https://" not in script
    assert "sessionStorage.getItem(TOKEN_KEY)" in script
    assert "LATENTSLATE_ENGINE_TOKEN" not in script
    assert "Authorization: Bearer " not in script
    assert 'method: "DELETE"' in script
    assert "delete_artifact" in script
    assert 'id="remove-declaration-button"' in index
    assert 'id="delete-resource-button"' in index
