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

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.policies.fineart_vla.processor_fineart_vla import SubtaskBoundaryHoldStep  # noqa: E402
from lerobot.processor import create_transition  # noqa: E402

FPS = 10.0


def _subtask(t: float, text: str) -> dict:
    return {"role": "assistant", "content": text, "style": "subtask", "timestamp": t, "camera": None}


def _transition(action, rows, timestamps):
    return create_transition(
        action=action,
        complementary_data={"language_persistent": rows, "timestamp": torch.tensor(timestamps)},
    )


def _ramp(batch: int, horizon: int = 8, dim: int = 3) -> torch.Tensor:
    return torch.arange(batch * horizon * dim, dtype=torch.float32).reshape(batch, horizon, dim)


def test_steps_after_the_subtask_end_hold_its_last_action():
    rows = [_subtask(0.0, "pick"), _subtask(1.0, "place")]
    action = _ramp(1)
    # t = 0.7 s: frames 0.7, 0.8, 0.9 are "pick"; 1.0 s onwards (step 3+) belongs to "place".
    out = SubtaskBoundaryHoldStep(enabled=True, fps=FPS)(_transition(action, [rows], [0.7]))
    held = out[TransitionKey.ACTION]
    torch.testing.assert_close(held[0, :3], action[0, :3])
    torch.testing.assert_close(held[0, 3:], action[0, 2].expand(5, -1))
    torch.testing.assert_close(action, _ramp(1), msg="input must not be modified in place")


def test_each_sample_uses_its_own_boundary():
    rows = [_subtask(0.0, "a"), _subtask(0.5, "b"), _subtask(2.0, "c")]
    action = _ramp(3)
    out = SubtaskBoundaryHoldStep(enabled=True, fps=FPS)(_transition(action, [rows] * 3, [0.4, 0.5, 0.9]))
    held = out[TransitionKey.ACTION]
    # 0.4 s: "b" starts one step later.
    torch.testing.assert_close(held[0, 1:], action[0, 0].expand(7, -1))
    # 0.5 s: sample starts exactly on "b"'s first frame; "c" (2.0 s) is beyond the 8-step chunk.
    torch.testing.assert_close(held[1], action[1])
    # 0.9 s: "c" is 11 steps away, beyond the chunk.
    torch.testing.assert_close(held[2], action[2])


def test_last_subtask_and_non_subtask_rows_are_left_alone():
    rows = [
        _subtask(0.0, "only"),
        {"role": "assistant", "content": "plan", "style": "plan", "timestamp": 0.3, "camera": None},
    ]
    action = _ramp(1)
    out = SubtaskBoundaryHoldStep(enabled=True, fps=FPS)(_transition(action, [rows], [0.1]))
    torch.testing.assert_close(out[TransitionKey.ACTION], action)


def test_unbatched_rows_are_accepted():
    rows = [_subtask(0.0, "pick"), _subtask(0.2, "place")]
    action = _ramp(1)
    out = SubtaskBoundaryHoldStep(enabled=True, fps=FPS)(_transition(action, rows, [0.0]))
    torch.testing.assert_close(out[TransitionKey.ACTION][0, 2:], action[0, 1].expand(6, -1))


def test_disabled_or_inference_transitions_pass_through():
    rows = [_subtask(0.0, "pick"), _subtask(0.1, "place")]
    action = _ramp(1)
    disabled = SubtaskBoundaryHoldStep(enabled=False, fps=FPS)(_transition(action, [rows], [0.0]))
    torch.testing.assert_close(disabled[TransitionKey.ACTION], action)
    # Inference: no action chunk, no fps needed.
    inference = create_transition(observation={}, complementary_data={"task": ["pick"]})
    assert SubtaskBoundaryHoldStep(enabled=True, fps=None)(inference) is inference


def test_missing_fps_fails_loudly_when_it_would_act():
    rows = [_subtask(0.0, "pick"), _subtask(0.1, "place")]
    with pytest.raises(ValueError, match="fps"):
        SubtaskBoundaryHoldStep(enabled=True, fps=None)(_transition(_ramp(1), [rows], [0.0]))


def test_config_roundtrip():
    step = SubtaskBoundaryHoldStep(enabled=True, fps=30.0)
    assert SubtaskBoundaryHoldStep(**step.get_config()).get_config() == {"enabled": True, "fps": 30.0}
