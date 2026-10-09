"""Static env-extension descriptor.

Lives in :mod:`parallelworld_manipulation.envs` alongside
:class:`~parallelworld_manipulation.envs.prompt_bundle.PromptBundle` so envs
and planners can both import it without pulling in
:mod:`parallelworld_manipulation.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`parallelworld_manipulation.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries the env identity, the prompt bundle, and runner hooks
that keep CLI orchestration independent of concrete robot implementations.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parallelworld_manipulation.dashboard.events import DashboardEventSink
from parallelworld_manipulation.envs.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from parallelworld_manipulation.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`EnvSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for parallelworld."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_shared_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    init_task_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    init_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    dashboard: dict[str, Any] | None = None
