"""The browser UI is served next to the API without getting in its way."""

import httpx


async def test_page_and_scripts_are_served(client: httpx.AsyncClient) -> None:
    page = await client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["cache-control"] == "no-cache"
    assert '<script type="module" src="ui/js/app.js">' in page.text

    script = await client.get("/ui/js/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert script.headers["cache-control"] == "no-cache"


async def test_missing_files_and_api_routes_stay_problems(client: httpx.AsyncClient) -> None:
    for path in ["/ui/nope.js", "/v1/nope"]:
        resp = await client.get(path)
        assert resp.status_code == 404
        assert resp.headers["content-type"] == "application/problem+json"
    assert "/" not in (await client.get("/openapi.json")).json()["paths"]
