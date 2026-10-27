"""Days 6-8 (stretch): physics playback, then PPO imitation in MuJoCo MJX.

Start from MuJoCo Playground's G1 environments (pip install playground) rather than writing
an MJX env from scratch. Reward = joint/pelvis tracking + upright bonus, weighted per phase
by motion_brief.reward_weights().
"""


def physics_playback(qpos_seq):
    """Day 6: PD-track the retargeted joint angles with physics on; return frames + where it fell."""
    raise NotImplementedError("Day 6")


def train(qpos_seq, phase_weights):
    """Day 7: PPO imitation on the A100."""
    raise NotImplementedError("Day 7")
