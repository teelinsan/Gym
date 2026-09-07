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
from pathlib import Path

from nemo_gym import PARENT_DIR, WORKING_DIR, _resolve_under_cwd_or_install


def failures_path_for(output_fpath: Path) -> Path:
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")


def aggregate_metrics_path_for(output_fpath: Path) -> Path:
    """`results/rollouts.jsonl` -> `results/rollouts_aggregate_metrics.json`.

    Mirrors how rollout collection and reverification name the file they write, so consumers
    (e.g. `gym eval compare`) derive the same path the writers produced.
    """
    return output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")


def report_dir_for(run_fpath: str | Path, *, output_dirpath: str | None = None, subdir: str | None = None) -> Path:
    """Directory to write a report *about* the run at `run_fpath` into.

    `output_dirpath` (an `--output-dir` flag) wins when set, taken relative to the user's cwd. Otherwise the
    report lands in the run's own directory, resolved the way the run's files were resolved for reading (see
    :func:`nemo_gym._resolve_under_cwd_or_install`) so it sits next to the metrics file that was actually
    read rather than at a same-named path under the cwd. `subdir`, when given, is appended.

    The one case where writing diverges from reading is a relative `run_fpath` that matched *only* under the
    install root: on a wheel install that root is `site-packages`, so the report goes beside the cwd instead.
    That is why this exists rather than each caller doing `_resolve_under_cwd_or_install(...).parent`, which
    that helper's own docstring rules out for write targets.

    Shared by `gym eval compare` and `gym eval stat-test`, which report on the same run.
    """
    if output_dirpath:
        base = Path(output_dirpath)
        base = base if base.is_absolute() else Path.cwd() / base
    else:
        base = _run_dir_for_writing(run_fpath)
    return base / subdir if subdir else base


def _run_dir_for_writing(run_fpath: str | Path) -> Path:
    """`run_fpath`'s directory, backing out of the install root when it is not the user's own tree."""
    resolved = _resolve_under_cwd_or_install(run_fpath)
    # WORKING_DIR != PARENT_DIR means a non-editable (wheel) install, i.e. PARENT_DIR is site-packages.
    if not Path(run_fpath).is_absolute() and WORKING_DIR != PARENT_DIR and _is_within(resolved, PARENT_DIR):
        resolved = Path.cwd() / run_fpath
    return resolved.parent


def _is_within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())
