"""Keep hardware probes out of provisioning unit tests."""

import pytest

from imaginer.provisioning import llamacpp


@pytest.fixture(autouse=True)
def no_host_probes(monkeypatch):
    monkeypatch.setattr(llamacpp, "detect_cuda_arch", lambda: "")
    monkeypatch.setattr(llamacpp, "detect_driver_cuda_version", lambda: (13, 4))
