# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end execution of `gym eval compare`."""

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from nemo_gym.comparison.diff import compare_runs
from nemo_gym.comparison.loading import build_loaded_run, load_agg_metrics_file, resolve_agent_selections
from nemo_gym.comparison.report import write_reports
from nemo_gym.comparison.schema import ComparisonConfig, ComparisonResult
from nemo_gym.config_types import ConfigError
from nemo_gym.package_info import __version__
from nemo_gym.path_utils import report_dir_for
from nemo_gym.secret_utils import hide_secrets_in_overrides


def invoked_command() -> str:
    """The `gym eval compare` invocation, for provenance in the report."""
    return shlex.join(["gym", "eval", "compare", *hide_secrets_in_overrides(sys.argv[1:])])


def build_comparison_result(config: ComparisonConfig, command: str) -> ComparisonResult:
    """Load both sides, pick the agent(s), and assemble the full comparison."""
    baseline_file = load_agg_metrics_file(
        config.baseline_rollouts_jsonl_fpath,
        role="baseline",
        aggregate_metrics_fpath_override=config.baseline_aggregate_metrics_fpath,
    )
    candidate_files = [
        load_agg_metrics_file(
            fpath,
            role="candidate",
            index=index,
            aggregate_metrics_fpath_override=(
                config.candidate_aggregate_metrics_fpaths[index] if config.candidate_aggregate_metrics_fpaths else None
            ),
        )
        for index, fpath in enumerate(config.candidate_rollouts_jsonl_fpaths)
    ]

    selections, warnings, skipped = resolve_agent_selections(
        baseline_file,
        candidate_files,
        agent_name=config.agent_name,
        baseline_agent_name=config.baseline_agent_name,
        candidate_agent_names=config.candidate_agent_names,
    )

    comparisons = []
    for selection in selections:
        baseline_run = build_loaded_run(baseline_file, selection.baseline_agent)
        candidate_runs = [
            build_loaded_run(run_file, agent) for run_file, agent in zip(candidate_files, selection.candidate_agents)
        ]
        comparisons.append(compare_runs(baseline_run, candidate_runs))

    return ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nemo_gym_version=__version__,
        command=command,
        baseline=baseline_file,
        candidates=candidate_files,
        comparisons=comparisons,
        skipped_agents=skipped,
        # Run-level warnings only. Per-comparison observations stay on their `AgentComparison.notes`
        # so the report renders each one once, under the agent it applies to.
        warnings=warnings,
    )


def resolve_output_dir(config: ComparisonConfig) -> Path:
    """`--output-dir`, defaulting to the candidate run's own directory.

    The default resolves the candidate path the same way loading does, so the report lands next to the
    metrics file that was actually read rather than at a same-named path under the cwd -- except when that
    would write into the install root. See :func:`~nemo_gym.path_utils.report_dir_for`.
    """
    return report_dir_for(config.candidate_rollouts_jsonl_fpaths[-1], output_dirpath=config.output_dirpath)


def run_comparison(config: ComparisonConfig, command: str) -> Tuple[ComparisonResult, List[Path]]:
    """Build the comparison, write its report artifacts, and run the statistical test alongside it."""
    result = build_comparison_result(config, command)
    written = write_reports(result, resolve_output_dir(config), config.report_format)

    if not config.no_stats:
        from nemo_gym.global_config import maybe_get_global_config_dict
        from nemo_gym.statistical_tests.common import stat_test_from_config_dict

        try:
            # `config` wins for everything it declares; the rest carries the test's own flags.
            stats_config_dict = {**(maybe_get_global_config_dict() or {}), **config.model_dump()}
            stats_markdown = stat_test_from_config_dict(stats_config_dict, "compare")
        except (ConfigError, ValidationError) as e:
            # A side effect of a comparison that already succeeded and was written: report it,
            # never turn a good `gym eval compare` into a failure.
            print(f"Skipped the statistical test: {e}")
        else:
            # Fenced verbatim: the test's report is line-oriented plain text, which markdown
            # would otherwise reflow into a single paragraph.
            section = f"\n## Statistical test\n\n```\n{stats_markdown.strip()}\n```\n"
            for markdown_report in [path for path in written if path.suffix == ".md"]:
                with markdown_report.open("a", encoding="utf-8") as f:
                    f.write(section)

    return result, written
