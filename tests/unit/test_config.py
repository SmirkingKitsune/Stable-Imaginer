from pathlib import Path

import pytest
from pydantic import ValidationError

from imaginer.core.config import (
    ConfigError,
    LlamaCppBackendConfig,
    RoutingConfig,
    SdCppBackendConfig,
    Settings,
    load_settings,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"

MINIMAL = """
data_root: {root}
server:
  port: 9100
backends:
  llama:
    type: llamacpp
    base_url: http://127.0.0.1:8080/
  sd:
    type: sdcpp
    base_url: http://127.0.0.1:1234
    defaults: {{steps: 20, cfg_scale: 6.0}}
routing:
  chat: llama
  t2i: sd
"""


def _write(tmp_path: Path, text: str, name: str = "test.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "profile", sorted(p.stem for p in REPO_CONFIG.glob("*.yaml") if ".local" not in p.stem)
)
def test_shipped_profiles_validate(profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Profiles that link over the network need the token their headers ask for.
    monkeypatch.setenv("IMAGINER_SERVER__AUTH_TOKEN", "test-token")
    settings = load_settings(profile, config_dir=REPO_CONFIG)
    assert settings.profile == profile
    assert settings.backends


def test_load_by_path_and_parse_backends(tmp_path: Path) -> None:
    settings = load_settings(str(_write(tmp_path, MINIMAL.format(root=tmp_path / "d"))))
    assert settings.server.port == 9100
    llama = settings.backends["llama"]
    assert isinstance(llama, LlamaCppBackendConfig)
    assert llama.base_url == "http://127.0.0.1:8080"  # trailing slash stripped
    sd = settings.backends["sd"]
    assert isinstance(sd, SdCppBackendConfig)
    assert sd.defaults.cfg_scale == 6.0
    assert sd.concurrency == 1


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGINER_SERVER__PORT", "9200")
    monkeypatch.setenv("IMAGINER_SERVER__AUTH_TOKEN", "s3cret")
    settings = load_settings(str(_write(tmp_path, MINIMAL.format(root=tmp_path / "d"))))
    assert settings.server.port == 9200
    assert settings.server.auth_token is not None
    assert settings.server.auth_token.get_secret_value() == "s3cret"
    assert "s3cret" not in settings.model_dump_json()


def test_profile_by_name_from_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, MINIMAL.format(root=tmp_path / "d"), "myhost.yaml")
    monkeypatch.setenv("IMAGINER_PROFILE", "myhost")
    assert load_settings(config_dir=tmp_path).profile == "myhost"


def test_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings("nope", config_dir=tmp_path)


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "server:\n  prot: 1\n")
    with pytest.raises(ValidationError):
        load_settings(str(path))


def _llama(base_url: str = "http://x:1") -> LlamaCppBackendConfig:
    return LlamaCppBackendConfig(type="llamacpp", base_url=base_url)


def test_routing_to_unknown_backend_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown backend"):
        Settings(routing=RoutingConfig(chat="ghost"))


def test_routing_kind_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="needs an llm backend"):
        Settings(
            backends={"sd": SdCppBackendConfig(type="sdcpp", base_url="http://x:1")},
            routing=RoutingConfig(chat="sd"),
        )


def test_backend_names_are_env_friendly() -> None:
    with pytest.raises(ValidationError):
        Settings(backends={"my-llama": _llama()})


def test_base_url_must_be_http() -> None:
    with pytest.raises(ValidationError, match="http"):
        _llama("nas.local:8080")


def test_data_root_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGINER_TEST_ROOT", str(tmp_path))
    settings = Settings(data_root=Path("$IMAGINER_TEST_ROOT") / "data")
    assert settings.data_root == (tmp_path / "data").resolve()
    assert Settings(data_root=Path("~") / "x").data_root == (Path.home() / "x").resolve()


def test_default_database_url_lives_under_data_root(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    assert settings.database_url.startswith("sqlite+aiosqlite:///")
    assert settings.database_url.endswith(f"{tmp_path.resolve().as_posix()}/imaginer.sqlite3")


def test_api_key_comes_from_named_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    config = LlamaCppBackendConfig(
        type="llamacpp", base_url="http://x:1", api_key_env="IMAGINER_TEST_KEY"
    )
    assert config.api_key() is None
    monkeypatch.setenv("IMAGINER_TEST_KEY", "k")
    assert config.api_key() == "k"


def test_networked_profiles_need_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGINER_SERVER__AUTH_TOKEN", raising=False)
    with pytest.raises(ValidationError, match="needs a server token"):
        load_settings("nas", config_dir=REPO_CONFIG)
    assert not load_settings("standalone", config_dir=REPO_CONFIG).links.enabled
