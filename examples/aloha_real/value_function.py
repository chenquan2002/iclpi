from __future__ import annotations

from typing import Any


def _pick_reward(step: dict[str, Any], reward_key: str | None) -> float:
    """Select a reward-like signal from rollout_info step metadata.

    Priority:
    1. Explicit reward_key if provided and present.
    2. returned_reward
    3. reward_diff
    4. reward
    5. 0.0 fallback
    """
    if reward_key is not None and reward_key in step:
        return float(step.get(reward_key, 0.0) or 0.0)

    for key in ("returned_reward", "reward_diff", "reward"):
        if key in step:
            return float(step.get(key, 0.0) or 0.0)
    return 0.0


def compute_value_labels(
    rollout_info: dict[str, Any],
    num_frames: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-step placeholder value labels from rollout sidecar metadata.

    This is a non-trained placeholder implementation. It reads rewards already
    saved during rollout and computes a discounted return-to-go:

        value_t = reward_t + gamma * value_{t+1}

    Then it uses the same scalar as a placeholder advantage:

        advantage_t = value_t

    Finally it creates a binary ACP-style indicator using a fixed threshold:

        indicator_t = 1 if advantage_t >= indicator_threshold else 0

    Args:
        rollout_info: Parsed episode sidecar, expected to contain a `steps` list.
        num_frames: Number of training samples for this episode. Must equal
            len(rollout_info['steps']).
        config: Optional config dict. Supported keys:
            - gamma: float, default 1.0
            - indicator_threshold: float, default 0.3
            - reward_key: str | None, explicit key to read from each step

    Returns:
        A list of length `num_frames`. Each element is a dict with keys:
            step_reward, step_reward_diff, acp_value, acp_advantage,
            acp_indicator
    """
    cfg = config or {}
    gamma = float(cfg.get("gamma", 1.0))
    indicator_threshold = float(cfg.get("indicator_threshold", 0.3))
    reward_key = cfg.get("reward_key")

    steps = rollout_info.get("steps", [])
    if len(steps) != num_frames:
        raise ValueError(
            f"compute_value_labels length mismatch: len(steps)={len(steps)} vs num_frames={num_frames}"
        )

    rewards = [_pick_reward(step, reward_key) for step in steps]
    reward_diffs = [float(step.get("reward_diff", 0.0) or 0.0) for step in steps]

    # Discounted return-to-go as placeholder value.
    values = [0.0 for _ in range(num_frames)]
    running = 0.0
    for i in range(num_frames - 1, -1, -1):
        running = rewards[i] + gamma * running
        values[i] = running

    # Placeholder advantage: directly reuse the value signal.
    advantages = list(values)
    indicators = [1 if adv >= indicator_threshold else 0 for adv in advantages]

    labels: list[dict[str, Any]] = []
    for i in range(num_frames):
        labels.append(
            {
                "step_reward": float(rewards[i]),
                "step_reward_diff": float(reward_diffs[i]),
                "acp_value": float(values[i]),
                "acp_advantage": float(advantages[i]),
                "acp_indicator": int(indicators[i]),
            }
        )
    return labels
