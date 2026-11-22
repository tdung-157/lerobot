# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The action-path decision must agree across ranks, or FSDP's all-gathers deadlock."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers")

from lerobot.policies.fineart_vla.modeling_fineart_vla import (  # noqa: E402
    predict_actions_any_across_ranks,
)


def _worker(rank: int, world_size: int, init_file: str, local_masks: list[list[bool]], results) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        results[rank] = predict_actions_any_across_ranks(torch.tensor(local_masks[rank]))
    finally:
        dist.destroy_process_group()


def _run(local_masks: list[list[bool]], tmp_path) -> list[bool]:
    world_size = len(local_masks)
    results = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, str(tmp_path / "pg_init"), local_masks, results),
        nprocs=world_size,
        join=True,
    )
    return [results[r] for r in range(world_size)]


@pytest.mark.parametrize(
    ("local_masks", "expected"),
    [
        # Rank 0 drew only text samples; rank 1 has an action sample: both must run the action path.
        ([[False, False], [False, True]], True),
        ([[True, False], [False, False]], True),
        # No rank needs actions: all ranks may skip the action expert together.
        ([[False, False], [False, False]], False),
    ],
)
def test_decision_is_or_reduced_across_ranks(local_masks, expected, tmp_path):
    assert _run(local_masks, tmp_path) == [expected, expected]


def test_single_process_keeps_the_local_decision():
    assert not dist.is_initialized()
    assert predict_actions_any_across_ranks(torch.tensor([False, False])) is False
    assert predict_actions_any_across_ranks(torch.tensor([False, True])) is True
    assert predict_actions_any_across_ranks(None) is True
