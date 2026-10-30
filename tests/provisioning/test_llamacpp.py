"""Unit tests for managed llama.cpp provisioning.

The plan builders and matrix helpers are pure, so everything here runs without
a toolchain, a GPU, or network access.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

from imaginer.provisioning.llamacpp import (
    UPDATE_BUILD_THRESHOLD,
    LlamaProvisioner,
    ProvisionConfig,
    ProvisionError,
    UpdateStatus,
    accelerator_cmake_flags,
    build_source_plan,
    builds_behind,
    check_for_update,
    choose_cuda_release,
    cuda_arch_targets,
    cuda_cmake_architectures,
    cuda_feature_target,
    installed_version_is_current,
    nvcc_supports_arch,
    parse_progress_fraction,
    read_active_runtime,
    release_accelerators,
    release_asset_patterns,
    release_build_number,
    select_release_assets,
    select_release_candidate,
    validate_server,
)


def config(**kwargs: Any) -> ProvisionConfig:
    """Config with the host probes pinned, so tests never touch hardware."""
    defaults: dict[str, Any] = {"platform": "linux", "arch": "x64", "accelerator": "cpu"}
    defaults.update(kwargs)
    return ProvisionConfig(**defaults)


class TestProvisionConfig(unittest.TestCase):
    def test_rejects_unknown_accelerator(self):
        with self.assertRaises(ProvisionError):
            config(accelerator="not-a-backend").normalized()

    def test_rejects_invalid_cuda_architecture(self):
        with self.assertRaises(ProvisionError):
            config(accelerator="cuda", cuda_arch="Blackwell").normalized()

    def test_force_reinstall_survives_normalization(self):
        self.assertTrue(config(force_reinstall=True).normalized().force_reinstall)


class TestCudaArchitectureMapping(unittest.TestCase):
    """Blackwell targets need the architecture-specific `a` spelling."""

    def test_blackwell_targets_gain_architecture_specific_suffix(self):
        # llama.cpp's MXFP4 kernels cannot assemble against a baseline 12X
        # target, so a detected sm_121 must be compiled as 121a.
        self.assertEqual(cuda_cmake_architectures("121"), "121a-real")
        self.assertEqual(cuda_cmake_architectures("120"), "120a-real")
        self.assertEqual(cuda_feature_target("121"), "121a")

    def test_pre_blackwell_targets_are_left_alone(self):
        self.assertEqual(cuda_cmake_architectures("86"), "86")
        self.assertEqual(cuda_cmake_architectures("90"), "90")
        self.assertEqual(cuda_feature_target("86"), "86")

    def test_targets_parse_from_every_accepted_spelling(self):
        self.assertEqual(cuda_arch_targets("sm_121"), ["121"])
        self.assertEqual(cuda_arch_targets("compute_86"), ["86"])
        self.assertEqual(cuda_arch_targets("121-real"), ["121"])
        self.assertEqual(cuda_arch_targets("86;90, 121"), ["86", "90", "121"])
        self.assertEqual(cuda_arch_targets("121 121"), ["121"])

    def test_empty_architecture_produces_no_cmake_flag(self):
        flags = accelerator_cmake_flags(config(accelerator="cuda", cuda_arch=""))
        self.assertEqual(flags, ["-DGGML_CUDA=ON"])

    def test_cuda_flags_carry_the_mapped_architecture(self):
        flags = accelerator_cmake_flags(config(accelerator="cuda", cuda_arch="121"))
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121a-real", flags)


class TestNvccArchSupport(unittest.TestCase):
    """`--list-gpu-arch` spells targets differently across toolkit versions."""

    def test_cuda_13_lists_only_the_base_architecture(self):
        # A 13.0 toolkit compiles sm_121a but lists only compute_121.
        self.assertTrue(nvcc_supports_arch("compute_121 compute_120", "121a"))

    def test_cuda_12_lists_the_architecture_specific_entry(self):
        self.assertTrue(nvcc_supports_arch("compute_90 compute_90a compute_120a", "120a"))

    def test_toolkit_too_old_lists_neither_spelling(self):
        self.assertFalse(nvcc_supports_arch("compute_70 compute_80 compute_90", "120a"))


class TestReleaseAssetMatrix(unittest.TestCase):
    LINUX_ASSETS: ClassVar[list[str]] = [
        "llama-b6000-bin-ubuntu-x64.tar.gz",
        "llama-b6000-bin-ubuntu-vulkan-x64.tar.gz",
        "llama-b6000-bin-ubuntu-sycl-fp16-x64.tar.gz",
        "llama-b6000-bin-ubuntu-sycl-fp32-x64.tar.gz",
        "llama-b6000-bin-ubuntu-cuda-12.4-x64.tar.gz",
        "llama-b6000-bin-ubuntu-cuda-13.1-x64.tar.gz",
    ]
    WINDOWS_ASSETS: ClassVar[list[str]] = [
        "llama-b6000-bin-win-cpu-x64.zip",
        "llama-b6000-bin-win-cuda-12.4-x64.zip",
        "llama-b6000-bin-win-cuda-13.1-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b6000-bin-win-vulkan-x64.zip",
    ]

    def test_linux_cpu_selects_the_plain_ubuntu_archive(self):
        selected = select_release_assets(self.LINUX_ASSETS, config(accelerator="cpu"))
        self.assertEqual(selected, ["llama-b6000-bin-ubuntu-x64.tar.gz"])

    def test_cuda_release_bounded_by_the_driver_version(self):
        selected = select_release_assets(
            self.LINUX_ASSETS, config(accelerator="cuda"), driver_version=(12, 6)
        )
        self.assertEqual(selected, ["llama-b6000-bin-ubuntu-cuda-12.4-x64.tar.gz"])

    def test_cuda_release_takes_the_newest_the_driver_allows(self):
        selected = select_release_assets(
            self.LINUX_ASSETS, config(accelerator="cuda"), driver_version=(13, 2)
        )
        self.assertEqual(selected, ["llama-b6000-bin-ubuntu-cuda-13.1-x64.tar.gz"])

    def test_unknown_driver_picks_the_lowest_cuda_build(self):
        choice = choose_cuda_release(
            ["llama-bin-cuda-12.4-x64.zip", "llama-bin-cuda-13.1-x64.zip"], None
        )
        self.assertEqual(choice, "llama-bin-cuda-12.4-x64.zip")

    def test_driver_older_than_every_build_has_no_release(self):
        with self.assertRaises(ProvisionError):
            select_release_assets(
                self.LINUX_ASSETS, config(accelerator="cuda"), driver_version=(11, 8)
            )

    def test_windows_cuda_layers_base_backend_and_cudart(self):
        selected = select_release_assets(
            self.WINDOWS_ASSETS,
            config(platform="windows", accelerator="cuda"),
            driver_version=(12, 6),
        )
        self.assertEqual(
            selected,
            [
                "llama-b6000-bin-win-cpu-x64.zip",
                "llama-b6000-bin-win-cuda-12.4-x64.zip",
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
            ],
        )

    def test_windows_cuda_requires_the_matching_cudart_archive(self):
        without_cudart = [name for name in self.WINDOWS_ASSETS if not name.startswith("cudart-")]
        with self.assertRaises(ProvisionError):
            select_release_assets(
                without_cudart,
                config(platform="windows", accelerator="cuda"),
                driver_version=(12, 6),
            )

    def test_release_variant_pins_the_cuda_major(self):
        selected = select_release_assets(
            self.LINUX_ASSETS,
            config(accelerator="cuda", release_variant="cuda-13"),
            driver_version=(13, 5),
        )
        self.assertEqual(selected, ["llama-b6000-bin-ubuntu-cuda-13.1-x64.tar.gz"])

    def test_arm64_linux_cuda_has_no_published_release(self):
        # The DGX Spark case: upstream publishes no aarch64 CUDA archive, which
        # is precisely why the source fallback has to exist.
        with self.assertRaises(ProvisionError):
            select_release_assets(
                ["llama-b6000-bin-ubuntu-cuda-12.4-x64.tar.gz"],
                config(arch="arm64", accelerator="cuda"),
            )

    def test_release_accelerators_reports_only_published_backends(self):
        available = release_accelerators(self.LINUX_ASSETS, config())
        self.assertIn("cuda", available)
        self.assertIn("vulkan", available)
        self.assertIn("sycl-fp32", available)
        self.assertIn("sycl-fp16", available)
        self.assertIn("cpu", available)
        self.assertNotIn("rocm", available)

    def test_macos_metal_uses_the_macos_archive(self):
        patterns = release_asset_patterns(config(platform="macos", accelerator="metal"))
        self.assertEqual(len(patterns), 1)
        self.assertIn("bin-macos", patterns[0])

    def test_latest_candidate_skips_metadata_only_stable_release(self):
        stable = {"tag_name": "v0.4.0", "assets": []}
        nightly = {
            "tag_name": "b6000",
            "assets": [{"name": self.LINUX_ASSETS[0]}],
        }
        release, selected = select_release_candidate([stable, nightly], config())
        self.assertEqual(release["tag_name"], "b6000")
        self.assertEqual(selected, [self.LINUX_ASSETS[0]])


class TestSourcePlan(unittest.TestCase):
    def test_cuda_plan_clears_the_stale_compiler_cache(self):
        plan = build_source_plan(config(accelerator="cuda", cuda_arch="121").normalized())
        labels = [step.label for step in plan]
        self.assertIn("Clearing cached CUDA compiler selection", labels)
        self.assertIn("Clearing cached CUDA configure state", labels)
        # Clearing must happen before configure, or the stale nvcc still wins.
        self.assertLess(
            labels.index("Clearing cached CUDA configure state"),
            labels.index("Configuring llama.cpp (CMake)"),
        )

    def test_cpu_plan_skips_the_cuda_cache_clearing(self):
        labels = [step.label for step in build_source_plan(config().normalized())]
        self.assertNotIn("Clearing cached CUDA compiler selection", labels)

    def test_configure_step_carries_the_mapped_cuda_architecture(self):
        plan = build_source_plan(config(accelerator="cuda", cuda_arch="121").normalized())
        configure = next(s for s in plan if s.label == "Configuring llama.cpp (CMake)")
        assert configure.argv is not None
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121a-real", configure.argv)
        self.assertIn("-DLLAMA_BUILD_SERVER=ON", configure.argv)

    def test_preflight_runs_first_and_is_skippable(self):
        plan = build_source_plan(config().normalized())
        self.assertEqual(plan[0].label, "Checking source-build prerequisites")

        bypassed = build_source_plan(config(bypass_environment_checks=True).normalized())
        self.assertNotEqual(bypassed[0].label, "Checking source-build prerequisites")

    def test_accelerator_gets_its_own_build_tree(self):
        cuda = config(accelerator="cuda").normalized().build_dir
        cpu = config(accelerator="cpu").normalized().build_dir
        self.assertNotEqual(cuda, cpu)
        self.assertIn("cuda", cuda.name)

    def test_gpu_builds_use_conservative_parallelism(self):
        # CUDA/ROCm translation units are memory-hungry enough to OOM.
        self.assertEqual(config(accelerator="cuda").normalized().effective_build_jobs, 2)
        self.assertEqual(config(accelerator="cpu").normalized().effective_build_jobs, 4)
        self.assertEqual(config(build_jobs=16).normalized().effective_build_jobs, 16)

    def test_pinned_version_checks_out_that_ref(self):
        plan = build_source_plan(config(version="b6000").normalized())
        labels = [s.label for s in plan]
        self.assertIn("Checking out b6000", labels)
        self.assertNotIn("Updating source", labels)

    def test_latest_tracks_master(self):
        plan = build_source_plan(config(version="latest").normalized())
        labels = [s.label for s in plan]
        self.assertIn("Checking out master", labels)
        self.assertIn("Updating source", labels)


class TestFallbackLadder(unittest.TestCase):
    """`auto` degrades in order rather than failing at the first obstacle."""

    @staticmethod
    def _labels(cfg: ProvisionConfig):
        return [label for label, _ in LlamaProvisioner(cfg)._attempts()]

    def test_published_backend_tries_release_then_source(self):
        labels = self._labels(config(accelerator="cuda"))
        self.assertEqual(labels[:2], ["matched release", "source fallback"])

    def test_unpublished_backend_still_reaches_the_source_build(self):
        # aarch64 Linux CUDA has no published asset today. The release rung is
        # still attempted rather than special-cased away: whether an asset
        # exists is only knowable from the API, and hardcoding today's gaps
        # would keep compiling from source after upstream fills them. The cost
        # of being wrong is one HTTP request.
        labels = self._labels(config(arch="arm64", accelerator="cuda"))
        self.assertIn("source fallback", labels)
        self.assertLess(labels.index("matched release"), labels.index("source fallback"))

    def test_backend_with_no_release_shape_skips_the_release_rung(self):
        # Metal has no Linux release shape, so there is nothing to request and
        # the ladder starts at the source build.
        labels = self._labels(config(accelerator="metal"))
        self.assertNotIn("matched release", labels)
        self.assertEqual(labels[0], "source fallback")

    def test_gpu_backends_keep_a_cpu_last_resort(self):
        # Restores the pre-port behaviour: a toolkit too old to either match a
        # release or compile one still leaves a working CPU runtime.
        self.assertEqual(self._labels(config(accelerator="cuda"))[-1], "cpu fallback")
        self.assertEqual(self._labels(config(arch="arm64", accelerator="cuda"))[-1], "cpu fallback")

    def test_cpu_target_has_no_redundant_fallback(self):
        self.assertNotIn("cpu fallback", self._labels(config(accelerator="cpu")))

    def test_explicit_method_disables_the_ladder(self):
        self.assertEqual(
            self._labels(config(accelerator="cuda", install_method="source")), ["source"]
        )
        self.assertEqual(
            self._labels(config(accelerator="cuda", install_method="release")), ["release"]
        )

    def test_cpu_fallback_drops_the_cuda_architecture(self):
        attempts = dict(LlamaProvisioner(config(accelerator="cuda", cuda_arch="121"))._attempts())
        self.assertEqual(attempts["cpu fallback"].cuda_arch, "")
        self.assertEqual(accelerator_cmake_flags(attempts["cpu fallback"]), [])


class TestActiveRuntimeState(unittest.TestCase):
    def test_missing_state_reports_nothing_installed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_active_runtime(Path(tmp)), {})

    def test_legacy_version_marker_is_still_reported(self):
        # Binaries installed before the JSON state file existed must not look
        # like a fresh machine.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            (bin_dir / "llama-server.version").write_text("b9082 (source build)", encoding="utf-8")
            active = read_active_runtime(Path(tmp))
            self.assertEqual(active["version"], "b9082 (source build)")
            self.assertEqual(active["method"], "legacy")


class TestVersionCurrency(unittest.TestCase):
    """Whether an installed runtime is already at least as new as a tag."""

    def test_exact_release_tag_is_current(self):
        self.assertTrue(installed_version_is_current("b10850", "b10850"))

    def test_source_build_containing_the_tag_is_current(self):
        self.assertTrue(installed_version_is_current("b10850-0-gabc1234 (source, cuda)", "b10850"))

    def test_older_source_build_is_not_current(self):
        self.assertFalse(
            installed_version_is_current("b10828-2-g465e49b9c (source, cuda)", "b10850")
        )

    def test_source_build_ahead_of_the_newest_tag_is_current(self):
        """A build from master routinely outruns the newest tagged release.

        String inequality alone would report an update available forever, and
        each "update" would recompile to something older.
        """
        self.assertTrue(installed_version_is_current("b10900-5-gabcdef0 (source, cuda)", "b10850"))

    def test_unparseable_version_is_not_assumed_current(self):
        self.assertFalse(installed_version_is_current("some-custom-build", "b10850"))

    def test_build_tag_prefix_is_not_assumed_current(self):
        self.assertFalse(installed_version_is_current("b1085", "b10850"))

    def test_matching_unparseable_versions_are_current(self):
        self.assertTrue(installed_version_is_current("custom-release", "custom-release"))

    def test_missing_inputs_are_not_current(self):
        self.assertFalse(installed_version_is_current("", "b10850"))
        self.assertFalse(installed_version_is_current("b10850", ""))

    def test_build_number_extraction(self):
        self.assertEqual(release_build_number("b10828-2-g465e49b9c"), 10828)
        self.assertEqual(release_build_number("b10850"), 10850)
        self.assertIsNone(release_build_number("no-build-number-here"))


class TestBuildDistance(unittest.TestCase):
    def test_exact_tag_is_zero(self):
        self.assertEqual(builds_behind("b10830", "b10830"), 0)
        self.assertEqual(builds_behind("b10830-0-gabc (source)", "b10830"), 0)

    def test_distance_is_counted_in_builds(self):
        self.assertEqual(builds_behind("b10700", "b10830"), 130)
        self.assertEqual(builds_behind("b10828-2-g465e49b9c", "b10830"), 2)

    def test_a_build_ahead_of_upstream_is_zero_not_negative(self):
        self.assertEqual(builds_behind("b10900-1-gabc", "b10830"), 0)

    def test_uncomparable_versions_return_none(self):
        self.assertIsNone(builds_behind("some-custom-build", "b10830"))
        self.assertIsNone(builds_behind("", "b10830"))


class TestUpdateThreshold(unittest.TestCase):
    """Upstream tags ~12 builds a day; without a floor the signal is noise."""

    def _check(self, installed: str, **kwargs: Any) -> UpdateStatus:
        releases = [{"tag_name": "b10830", "assets": TestUpdateCheck.ASSETS}]
        with patch("imaginer.provisioning.llamacpp.fetch_releases", return_value=releases):
            return check_for_update(config(accelerator="cuda"), installed, **kwargs)

    def test_a_few_builds_behind_is_not_reported(self):
        status = self._check("b10828-2-g465e49b9c (source, cuda)")
        self.assertEqual(status.behind, 2)
        self.assertFalse(status.update_available)
        self.assertTrue(status.below_threshold)
        self.assertFalse(status.is_current)

    def test_far_behind_is_reported(self):
        status = self._check("b10700")
        self.assertEqual(status.behind, 130)
        self.assertTrue(status.update_available)
        self.assertFalse(status.below_threshold)

    def test_exactly_at_the_threshold_is_reported(self):
        status = self._check("b10730")  # 100 behind
        self.assertEqual(status.behind, UPDATE_BUILD_THRESHOLD)
        self.assertTrue(status.update_available)

    def test_one_below_the_threshold_is_not(self):
        status = self._check("b10731")  # 99 behind
        self.assertFalse(status.update_available)

    def test_current_runtime_is_neither_behind_nor_below_threshold(self):
        status = self._check("b10830")
        self.assertTrue(status.is_current)
        self.assertFalse(status.below_threshold)
        self.assertFalse(status.update_available)

    def test_threshold_zero_reports_any_difference(self):
        status = self._check("b10828-2-g465e49b9c (source, cuda)", threshold=0)
        self.assertTrue(status.update_available)
        self.assertEqual(status.behind, 2)

    def test_uncomparable_version_is_reported_rather_than_hidden(self):
        # Failing to parse a custom build string is not evidence it is current;
        # suppressing the update there would hide a real one behind a threshold
        # that was never actually evaluated.
        status = self._check("my-hand-rolled-llama-server")
        self.assertIsNone(status.behind)
        self.assertTrue(status.update_available)


class TestUpdateCheck(unittest.TestCase):
    """check_for_update reports the cost of an update, not just its existence."""

    # Mirrors what upstream actually publishes: x64 gets a CUDA archive,
    # arm64 gets CPU and Vulkan but never CUDA.
    ASSETS: ClassVar[list[dict[str, str]]] = [
        {"name": "llama-b10850-bin-ubuntu-x64.tar.gz"},
        {"name": "llama-b10850-bin-ubuntu-vulkan-x64.tar.gz"},
        {"name": "llama-b10850-bin-ubuntu-cuda-12.4-x64.tar.gz"},
        {"name": "llama-b10850-bin-ubuntu-arm64.tar.gz"},
        {"name": "llama-b10850-bin-ubuntu-vulkan-arm64.tar.gz"},
    ]

    def _check(
        self, cfg: ProvisionConfig, installed: str, releases: list[dict[str, Any]] | None = None
    ) -> UpdateStatus:
        payload = (
            releases if releases is not None else [{"tag_name": "b10850", "assets": self.ASSETS}]
        )
        # threshold=0: these cases are about how an update would be obtained,
        # not about whether it is far enough behind to mention. The staleness
        # floor has its own tests.
        with patch("imaginer.provisioning.llamacpp.fetch_releases", return_value=payload):
            return check_for_update(cfg, installed, threshold=0)

    def test_current_runtime_reports_no_update(self):
        status = self._check(config(accelerator="cuda"), "b10850")
        self.assertFalse(status.update_available)
        self.assertEqual(status.latest_version, "b10850")
        self.assertTrue(status.ok)

    def test_published_backend_updates_via_release(self):
        status = self._check(config(accelerator="cuda"), "b10800")
        self.assertTrue(status.update_available)
        self.assertEqual(status.action, "release")
        self.assertEqual(status.warning, "")

    def test_unpublished_backend_requires_a_rebuild_and_says_so(self):
        # The DGX Spark case: an update exists but costs twenty minutes.
        status = self._check(config(arch="arm64", accelerator="cuda"), "b10800")
        self.assertTrue(status.update_available)
        self.assertEqual(status.action, "compile")
        self.assertIn("compile llama-server from source", status.warning)

    def test_release_only_method_reports_an_unavailable_update(self):
        status = self._check(
            config(arch="arm64", accelerator="cuda", install_method="release"), "b10800"
        )
        self.assertEqual(status.action, "unavailable")
        self.assertIn("restricted to releases", status.warning)

    def test_alternatives_are_offered_when_the_target_has_no_release(self):
        status = self._check(config(arch="arm64", accelerator="cuda"), "b10800")
        self.assertIn("cpu", status.release_alternatives)
        self.assertIn("Official releases do exist", status.warning)

    def test_source_install_method_always_compiles(self):
        # Even where a release exists: switching a source install to a release
        # binary silently discards a build tuned for this machine.
        status = self._check(config(accelerator="cuda", install_method="source"), "b10800")
        self.assertEqual(status.action, "compile")

    def test_cuda_release_too_new_for_driver_requires_compile(self):
        with patch(
            "imaginer.provisioning.llamacpp.detect_driver_cuda_version",
            return_value=(11, 8),
        ):
            status = self._check(config(accelerator="cuda"), "b10800")
        self.assertEqual(status.action, "compile")
        self.assertIn("compile llama-server from source", status.warning)

    def test_network_failure_is_reported_not_raised(self):
        with patch(
            "imaginer.provisioning.llamacpp.fetch_releases", side_effect=OSError("no route to host")
        ):
            status = check_for_update(config(accelerator="cuda"), "b10800")
        self.assertFalse(status.ok)
        self.assertIn("no route to host", status.error)
        self.assertFalse(status.update_available)

    def test_empty_release_list_is_an_error_not_an_update(self):
        status = self._check(config(accelerator="cuda"), "b10800", releases=[])
        self.assertFalse(status.ok)
        self.assertFalse(status.update_available)

    def test_metadata_only_release_is_skipped(self):
        status = self._check(
            config(accelerator="cuda"),
            "b10800",
            releases=[
                {"tag_name": "v0.4.0", "assets": []},
                {"tag_name": "b10850", "assets": self.ASSETS},
            ],
        )
        self.assertTrue(status.ok)
        self.assertEqual(status.latest_version, "b10850")
        self.assertEqual(status.action, "release")

    def test_release_list_without_a_build_tag_is_an_error(self):
        status = self._check(
            config(accelerator="cuda"), "b10800", releases=[{"tag_name": "", "assets": []}]
        )
        self.assertFalse(status.ok)


class TestProgressParsing(unittest.TestCase):
    def test_cmake_percentage_becomes_a_fraction(self):
        self.assertAlmostEqual(parse_progress_fraction("[ 42%] Building CXX object"), 0.42)
        self.assertAlmostEqual(parse_progress_fraction("[100%] Linking"), 1.0)

    def test_unrelated_output_reports_no_progress(self):
        self.assertEqual(parse_progress_fraction("-- Configuring done"), -1.0)

    def test_runtime_validation_accepts_a_working_executable(self):
        detail = validate_server(Path(sys.executable))
        self.assertIn("Python", detail)


if __name__ == "__main__":
    unittest.main()
