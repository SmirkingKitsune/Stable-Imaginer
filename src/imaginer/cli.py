"""``imaginer`` command line. A Python entrypoint, so it behaves the same on Windows and Linux.

imaginer serve --profile windows-5090
imaginer backends check --profile windows-5090
imaginer db upgrade --profile nas
imaginer config show --profile dev
imaginer link code --profile nas
"""

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from imaginer.core.config import ConfigError, Settings, load_settings


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--profile",
        help="config profile name (config/<name>.yaml) or a YAML path; "
        "default: $IMAGINER_PROFILE, then 'dev'",
    )
    common.add_argument("--config-dir", type=Path, help="directory holding profile YAML files")

    parser = argparse.ArgumentParser(prog="imaginer", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", parents=[common], help="run the HTTP server")
    serve.add_argument("--host", help="override server.host")
    serve.add_argument("--port", type=int, help="override server.port")

    db = sub.add_parser("db", help="database maintenance")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("upgrade", parents=[common], help="apply pending migrations")
    db_sub.add_parser("check", parents=[common], help="fail if models and migrations differ")

    backends = sub.add_parser("backends", help="engine backends")
    backends_sub = backends.add_subparsers(dest="backends_command", required=True)
    backends_sub.add_parser(
        "check", parents=[common], help="probe every backend; exit 1 if an enabled one is not ok"
    )

    config = sub.add_parser("config", help="inspect configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", parents=[common], help="print the resolved config")

    link = sub.add_parser("link", help="links with other instances (docs/linked-instances.md)")
    link_sub = link.add_subparsers(dest="link_command", required=True)
    link_sub.add_parser(
        "code", parents=[common], help="make a pairing code another instance can link with"
    )
    link_sub.add_parser("list", parents=[common], help="list links and whether they answer")
    link_add = link_sub.add_parser(
        "add", parents=[common], help="link with the instance at URL, using the code it shows"
    )
    link_add.add_argument("url", help="e.g. http://nas.local:8700")
    link_add.add_argument("--code", required=True, help="the pairing code shown there")
    link_add.add_argument("--name", help="this instance's name for the link")
    link_remove = link_sub.add_parser("remove", parents=[common], help="unlink")
    link_remove.add_argument("name", help="the link's name, as `imaginer link list` shows it")
    from imaginer.provisioning import llamacpp, sdcpp

    for name, engine, folder in (
        ("llama", llamacpp, "llama_cpp"),
        ("sdcpp", sdcpp, "stable_diffusion_cpp"),
    ):
        runtime_parser = sub.add_parser(name, help=f"provision and inspect {name}")
        runtime_sub = runtime_parser.add_subparsers(dest="runtime_command", required=True)
        for command in ("provision", "plan", "status", "check-update"):
            runtime = runtime_sub.add_parser(command, parents=[common])
            runtime.add_argument(
                "--cache-dir", type=Path, help=f"default: <data_root>/engines/{folder}"
            )
            runtime.add_argument("--method", choices=engine.INSTALL_METHODS, default="auto")
            runtime.add_argument("--tag", default="latest", help="release tag or source git ref")
            runtime.add_argument("--accelerator", choices=engine.ACCELERATORS, default="")
            runtime.add_argument("--cuda-arch", default="", help="e.g. 120 or 121; auto-detected")
            if name == "llama":
                runtime.add_argument(
                    "--release-variant", choices=("cuda-12", "cuda-13"), default=""
                )
            else:
                runtime.set_defaults(release_variant="")
            runtime.add_argument(
                "--jobs", type=int, default=0, help="0: conservative build default"
            )
            runtime.add_argument(
                "--cmake-arg", action="append", default=[], help="use --cmake-arg=-D..."
            )
            runtime.add_argument("--force", action="store_true", help="replace an existing runtime")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv(override=False)  # a local, gitignored .env may hold secrets
    if args.command in ("llama", "sdcpp"):
        return _provision_engine(args)
    try:
        settings = load_settings(args.profile, config_dir=args.config_dir)
    except (ConfigError, ValidationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "serve":
        return _serve(settings, args.host, args.port)
    if args.command == "db":
        return _db(settings, args.db_command)
    if args.command == "backends":
        return asyncio.run(_check_backends(settings))
    if args.command == "config":
        print(json.dumps(settings.model_dump(mode="json"), indent=2))
        return 0
    if args.command == "link":
        return asyncio.run(_link(settings, args))
    return 2


def _provision_engine(args: argparse.Namespace) -> int:
    """Installation is explicit; serving an HTTP backend never installs an engine."""
    from dataclasses import asdict, replace

    from imaginer.provisioning import llamacpp, sdcpp
    from imaginer.provisioning.llamacpp import ProvisionError

    engine = sdcpp if args.command == "sdcpp" else llamacpp
    runner = sdcpp.SdCppProvisioner if args.command == "sdcpp" else llamacpp.LlamaProvisioner
    folder = "stable_diffusion_cpp" if args.command == "sdcpp" else "llama_cpp"

    provisioner = None
    try:
        settings = (
            load_settings(args.profile, config_dir=args.config_dir)
            if args.profile or args.config_dir
            else Settings()
        )
        root = args.cache_dir or settings.data_root / "engines" / folder
        if args.runtime_command == "status":
            active = engine.read_active_runtime(root)
            executable = active.get("executable")
            print(
                json.dumps(
                    {**active, "installed": bool(executable and Path(executable).is_file())},
                    indent=2,
                )
            )
            return 0
        config = engine.ProvisionConfig(
            provision_dir=root,
            install_method=args.method,
            version=args.tag,
            accelerator=args.accelerator,
            cuda_arch=args.cuda_arch,
            release_variant=args.release_variant,
            build_jobs=args.jobs,
            cmake_args=args.cmake_arg,
            force_reinstall=args.force,
        ).normalized()
        provisioner = runner(
            config, log_sink=lambda line, is_stderr: print(line, file=sys.stderr, flush=True)
        )
        if args.runtime_command == "plan":
            plans = []
            for label, attempt in provisioner.install_attempts():
                steps = engine.build_install_plan(attempt)
                plans.append(
                    {
                        "attempt": label,
                        "accelerator": attempt.accelerator,
                        "steps": [
                            {
                                "label": step.label,
                                "argv": step.argv,
                                "cwd": str(step.cwd) if step.cwd is not None else None,
                            }
                            for step in steps
                        ],
                    }
                )
            print(json.dumps(plans, indent=2))
            return 0
        if args.runtime_command == "check-update":
            active = engine.read_active_runtime(config.provision_dir)
            config = replace(
                config,
                accelerator=args.accelerator or str(active.get("accelerator", config.accelerator)),
                install_method=(
                    str(active.get("method", "auto")) if args.method == "auto" else args.method
                ),
            )
            status = engine.check_for_update(config, str(active.get("version", "")))
            print(json.dumps(asdict(status), indent=2))
            return 0 if status.ok else 1
        binary = provisioner.ensure_runtime()
        print(
            json.dumps(
                {
                    **engine.read_active_runtime(config.provision_dir),
                    "executable": str(binary),
                    "used_existing": provisioner.used_existing,
                    "warning": provisioner.target_mismatch,
                },
                indent=2,
            )
        )
        return 0
    except (ConfigError, ValidationError, ProvisionError, OSError, ValueError) as exc:
        print(f"{runner.engine_name} provisioning error: {exc}", file=sys.stderr)
        if provisioner and provisioner.build_log_path:
            print(f"build log: {provisioner.build_log_path}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"{runner.engine_name} provisioning cancelled", file=sys.stderr)
        return 130


def _serve(settings: Settings, host: str | None, port: int | None) -> int:
    import uvicorn

    from imaginer.main import create_app

    updates: dict[str, object] = {}
    if host:
        updates["host"] = host
    if port:
        updates["port"] = port
    if updates:
        settings.server = settings.server.model_copy(update=updates)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,  # keep our structlog setup
        access_log=False,  # RequestContextMiddleware logs requests with their ids
    )
    return 0


def _db(settings: Settings, command: str) -> int:
    from alembic.util import AutogenerateDiffsDetected, CommandError

    from imaginer.db import migrate

    settings.data_root.mkdir(parents=True, exist_ok=True)
    url = settings.database_url
    if command == "upgrade":
        migrate.upgrade(url)
        print(f"database at head: {url}")
        return 0
    try:
        migrate.check(url)
    except AutogenerateDiffsDetected as exc:
        print(f"models and migrations differ: {exc}", file=sys.stderr)
        return 1
    except CommandError as exc:  # e.g. the database was never migrated
        print(f"cannot check: {exc} (run `imaginer db upgrade` first)", file=sys.stderr)
        return 1
    print("models and migrations agree")
    return 0


async def _link(settings: Settings, args: argparse.Namespace) -> int:
    """Link commands work on the database directly, so they also work on a headless host
    whose server is running: it picks the changes up on its next poll."""
    from imaginer.core.logging import configure_logging
    from imaginer.domain import models as dto
    from imaginer.domain.errors import ImaginerError

    configure_logging("WARNING", settings.logging.format)
    async with _links_service(settings) as links:
        try:
            if args.link_command == "code":
                code = await links.mint_code()
                expires = code.expires_at.astimezone().strftime("%H:%M")
                print(f"pairing code: {code.code}  (valid until {expires}, single use)")
                print(f"on the other instance, link with {links.advertise_url} and this code")
            elif args.link_command == "add":
                made = await links.pair_with(
                    dto.LinkCreate(url=args.url, code=args.code, name=args.name)
                )
                print(f"linked with {made.peer_name} as {made.name!r} ({made.status})")
            elif args.link_command == "remove":
                found = links.by_name(args.name)
                if found is None:
                    print(f"no link named {args.name!r}", file=sys.stderr)
                    return 1
                await links.unlink(found.id)
                print(f"unlinked {found.peer_name}")
            else:
                await asyncio.gather(*(links.poll(item) for item in links.all()))
                if not links.all():
                    print("no links")
                for item in links.all():
                    shared = ", ".join(item.engines) or "-"
                    print(
                        f"{item.name:<16} {item.peer_name:<20} {item.status:<12} {item.url}"
                        f"  shares: {shared}"
                    )
        except ImaginerError as exc:
            print(f"{exc.code}: {exc.detail}", file=sys.stderr)
            return 1
    return 0


@contextlib.asynccontextmanager
async def _links_service(settings: Settings) -> AsyncIterator[Any]:
    from imaginer.db import migrate
    from imaginer.db.engine import Database
    from imaginer.links.service import LinkService

    settings.data_root.mkdir(parents=True, exist_ok=True)
    if settings.db.auto_migrate:
        await asyncio.to_thread(migrate.upgrade, settings.database_url)
    db = Database(settings.database_url)
    links = LinkService(settings, db)
    try:
        await links.start(background=False)
        yield links
    finally:
        await links.aclose()
        await db.dispose()


async def _check_backends(settings: Settings) -> int:
    from imaginer.backends.registry import FACTORIES, BackendRegistry
    from imaginer.core.logging import configure_logging
    from imaginer.domain.enums import BackendStatus

    configure_logging("WARNING", settings.logging.format)  # keep the table readable
    async with _links_service(settings) as links:
        registry = BackendRegistry.from_settings(
            settings, factories={**FACTORIES, "linked": links.backend_factory}
        )
        try:
            infos = await registry.describe()
        finally:
            await registry.aclose()
    if not infos:
        print("no backends configured")
        return 0
    width = max(len(i.name) for i in infos)
    healthy = True
    for info in infos:
        caps = ",".join(info.capabilities) or "-"
        latency = f"{info.latency_ms:.0f}ms" if info.latency_ms is not None else ""
        print(
            f"{info.name:<{width}}  {info.type:<9} {info.status.value:<11} {latency:>7}  "
            f"caps={caps}  {info.detail or ''}"
        )
        if info.enabled and info.status is not BackendStatus.OK:
            healthy = False
    return 0 if healthy else 1
