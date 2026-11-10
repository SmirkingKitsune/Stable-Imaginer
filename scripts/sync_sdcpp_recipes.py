"""Refresh the bundled snapshot of sd-server's built-in recipes.

stable-diffusion.cpp's sd-server ships one JSON recipe per model family
(``examples/server/recipes/*.json``): the component slots a family needs, its generation
defaults, size multiple and features. Stable-Imaginer reads them to know how to launch and
drive each image family (src/imaginer/image_models/recipes.py). A running engine's own
recipes (``GET /sdcpp/v1/recipes``) take precedence over this snapshot.

    uv run python scripts/sync_sdcpp_recipes.py            # master
    uv run python scripts/sync_sdcpp_recipes.py --ref 37bc88b

Video recipes are left out: Stable-Imaginer makes images only.
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = "leejet/stable-diffusion.cpp"
FOLDER = "examples/server/recipes"
OUT = Path(__file__).resolve().parents[1] / "src/imaginer/image_models/data/sdcpp_recipes.json"
# Slots and features only video (or audio) pipelines use.
VIDEO_SLOTS = frozenset(
    {"high_noise_diffusion_model", "audio_vae", "audio_encoder", "motion_module"}
)
VIDEO_FEATURES = frozenset({"ref_videos", "ref_audios", "ref_video_audios", "high_noise"})


def _get(url: str, accept: str = "application/json", attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "stable-imaginer-recipe-sync"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data: bytes = response.read()
                return data
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def _is_video(recipe: dict[str, Any]) -> bool:
    if "video_constraints" in recipe:
        return True
    defaults = recipe.get("defaults") or {}
    if any(key.startswith("video") for key in defaults):
        return True
    slots = {slot.get("id") for slot in recipe.get("slots") or []}
    return bool(slots & VIDEO_SLOTS or set(recipe.get("features") or []) & VIDEO_FEATURES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="master", help="branch, tag or commit")
    args = parser.parse_args()

    commit = _get(
        f"https://api.github.com/repos/{REPO}/commits/{args.ref}",
        accept="application/vnd.github.sha",
    ).decode()
    listing = json.loads(
        _get(f"https://api.github.com/repos/{REPO}/contents/{FOLDER}?ref={commit}")
    )
    recipes = []
    for entry in sorted(listing, key=lambda e: str(e["name"])):
        name = str(entry["name"])
        if not name.endswith(".json"):
            continue
        raw = _get(f"https://raw.githubusercontent.com/{REPO}/{commit}/{FOLDER}/{name}")
        recipe = json.loads(raw)
        if _is_video(recipe):
            continue
        recipes.append(recipe)
    snapshot = {
        "source": f"https://github.com/{REPO}/tree/{commit}/{FOLDER}",
        "commit": commit,
        "fetched": datetime.now(UTC).date().isoformat(),
        "recipes": recipes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(recipes)} image recipes from {commit[:7]} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
