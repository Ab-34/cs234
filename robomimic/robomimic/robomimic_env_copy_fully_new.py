"""
RobomimicEnvWrapper
===================
A Gym-compatible wrapper around robosuite's Lift task for use with
Stable Baselines 3 (PPO).

Design goals for BC → PPO stability
--------------------------------------
1. Dense reward shaping  — sparse {0,1} reward starves PPO early on; we add
   distance-to-cube and gripper-alignment bonuses so the value function has
   gradient signal from step 0.
2. Consistent observation flattening — the wrapper guarantees the same obs
   vector order every reset/step, matching what the BC-pretrained policy saw.
3. Action rescaling — robosuite actions are in [-1, 1]; the wrapper enforces
   this so the BC policy's learned scale assumptions are preserved.
4. Stable termination — episode ends on success OR horizon, never silently
   truncates mid-step.
5. Numpy-seed support — pass `seed` to reset() for reproducible evals.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, List

import robosuite as suite
# from robosuite import load_controller_config
from robosuite import load_composite_controller_config



# ─────────────────────────────────────────────────────────────────────────────
# Observation key groups (order matters — must match BC training data)
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_OBS_KEYS: List[str] = [
    "robot0_joint_pos",       # (7,)  joint angles
    "robot0_joint_vel",       # (7,)  joint velocities
    "robot0_eef_pos",         # (3,)  end-effector xyz
    "robot0_eef_quat",        # (4,)  end-effector quaternion
    "robot0_gripper_qpos",    # (2,)  gripper finger positions
    "robot0_gripper_qvel",    # (2,)  gripper finger velocities
]

OBJECT_OBS_KEYS: List[str] = [
    "cube_pos",               # (3,)  cube position
    "cube_quat",              # (4,)  cube orientation
    "gripper_to_cube_pos",    # (3,)  vector from gripper → cube
    "cube_to_goal_pos",       # (3,)  vector from cube → goal (for place tasks)
]

# Keys we *try* to use but skip gracefully if absent (e.g. fixed camera setups)
OPTIONAL_KEYS: List[str] = ["cube_to_goal_pos"]


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping weights  (tune these; set all to 0 for pure sparse)
# ─────────────────────────────────────────────────────────────────────────────

class RewardConfig:
    sparse_success:      float = 2.0   # +2 on task completion
    dist_reach:          float = 1.0   # reward for EEF approaching cube
    grasp_bonus:         float = 0.5   # reward when gripper closes on cube
    lift_height:         float = 1.0   # reward proportional to cube height
    action_penalty:      float = 0.01  # small penalty for large actions (smoothness)


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RobomimicEnvWrapper(gym.Env):
    """
    Wraps robosuite's Lift task to be compatible with SB3's PPO.

    Parameters
    ----------
    env_name        : robosuite task name (default "Lift")
    robots          : robot arm (default "Panda")
    use_camera_obs  : whether to include camera observations (default False)
    render_mode     : "rgb_array" or "human"
    has_renderer    : open a viewer window (default False for headless training)
    use_object_obs  : include object state in observation (default True)
    horizon         : max steps per episode
    reward_shaping  : if True, use dense reward; if False, use sparse only
    full_loss       : unused flag kept for CLI argument compatibility
    seed            : optional RNG seed
    controller      : robosuite controller type
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env_name:       str  = "Lift",
        robots:         str  = "Panda",
        use_camera_obs: bool = False,
        render_mode:    str  = "rgb_array",
        has_renderer:   bool = False,
        use_object_obs: bool = True,
        horizon:        int  = 500,
        reward_shaping: bool = True,
        full_loss:      bool = False,
        seed:           Optional[int] = None,
        controller:     str  = "OSC_POSE",
    ):
        super().__init__()

        self.env_name       = env_name
        self.robots         = robots
        self.use_camera_obs = use_camera_obs
        self.render_mode    = render_mode
        self.has_renderer   = has_renderer
        self.use_object_obs = use_object_obs
        self.horizon        = horizon
        self.reward_shaping = reward_shaping
        self.full_loss      = full_loss
        self.seed_val       = seed
        self.reward_cfg     = RewardConfig()

        # ── Build robosuite env ───────────────────────────────────────────────
        controller_config = load_composite_controller_config(default_controller=controller)

        self._env = suite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=(render_mode == "rgb_array"),
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_shaping=False,    # we do our own reward shaping below
            horizon=horizon,
            ignore_done=False,
            hard_reset=False,        # faster resets during training
        )

        # ── Introspect observation structure ─────────────────────────────────
        self._obs_keys, obs_dim = self._resolve_obs_keys()
        print(f"[RobomimicEnvWrapper] obs_dim={obs_dim}  keys={self._obs_keys}")

        # ── Gym spaces ───────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        act_dim = self._env.action_spec[0].shape[0]  # typically 7
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # ── Episode bookkeeping ───────────────────────────────────────────────
        self._step_count  = 0
        self._last_raw_obs: Optional[Dict] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_obs_keys(self) -> Tuple[List[str], int]:
        """
        Do a throwaway reset to discover which observation keys exist and
        their dimensions.  Returns (ordered_key_list, total_flat_dim).
        """
        raw = self._env.reset()
        desired = ROBOT_OBS_KEYS + (OBJECT_OBS_KEYS if self.use_object_obs else [])
        present = []
        total   = 0
        for k in desired:
            if k in raw:
                present.append(k)
                total += int(np.prod(raw[k].shape))
            elif k not in OPTIONAL_KEYS:
                print(f"  [WARN] Expected obs key '{k}' not found — skipping.")
        return present, total

    def _flatten_obs(self, raw_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict-obs into a fixed-order 1-D float32 vector."""
        parts = []
        for k in self._obs_keys:
            v = raw_obs[k]
            parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts, axis=0)

    def _compute_reward(
        self,
        raw_obs:    Dict[str, np.ndarray],
        raw_reward: float,
        action:     np.ndarray,
    ) -> float:
        """
        Dense reward on top of the sparse robosuite signal.

        Breakdown:
          sparse_success   — +2.0 when task completed
          dist_reach       — Gaussian-style bonus for EEF → cube proximity
          grasp_bonus      — bonus proportional to gripper closure on cube
          lift_height      — cube z-position relative to table
          action_penalty   — penalise jerk / large actions for smooth policy
        """
        if not self.reward_shaping:
            return float(raw_reward)

        r   = self.reward_cfg
        rew = 0.0

        # 1. Sparse success
        rew += r.sparse_success * float(raw_reward)

        # 2. Reach: distance from EEF to cube
        if "robot0_eef_pos" in raw_obs and "cube_pos" in raw_obs:
            eef_pos  = raw_obs["robot0_eef_pos"]
            cube_pos = raw_obs["cube_pos"]
            dist     = float(np.linalg.norm(eef_pos - cube_pos))
            # Exponential reach bonus — peaks at 1.0 when dist≈0
            rew += r.dist_reach * np.exp(-5.0 * dist)

        # 3. Grasp bonus: gripper closing (qpos → 0 means closed for Panda)
        if "robot0_gripper_qpos" in raw_obs:
            gripper_q = raw_obs["robot0_gripper_qpos"]
            # Panda finger range ≈ [0, 0.04]; closed ≈ 0
            closure   = 1.0 - float(np.mean(np.abs(gripper_q)) / 0.04)
            rew      += r.grasp_bonus * np.clip(closure, 0.0, 1.0)

        # 4. Lift height: reward cube being above table (~0.81 m in world frame)
        if "cube_pos" in raw_obs:
            table_z   = 0.81          # approximate table surface in world frame
            cube_z    = float(raw_obs["cube_pos"][2])
            lift      = np.clip(cube_z - table_z, 0.0, 0.20) / 0.20
            rew      += r.lift_height * lift

        # 5. Action smoothness penalty
        rew -= r.action_penalty * float(np.sum(np.square(action)))

        return rew

    # ─────────────────────────────────────────────────────────────────────────
    # Gym API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return (obs, info)."""
        if seed is not None:
            np.random.seed(seed)
        elif self.seed_val is not None:
            np.random.seed(self.seed_val)

        raw_obs           = self._env.reset()
        self._last_raw_obs = raw_obs
        self._step_count  = 0

        obs  = self._flatten_obs(raw_obs)
        info = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Returns (obs, reward, terminated, truncated, info).
        terminated = task success
        truncated  = horizon exceeded
        """
        # Clip action to valid range (important if PPO samples outside [-1,1])
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        raw_obs, raw_reward, done, info = self._env.step(action)
        self._last_raw_obs  = raw_obs
        self._step_count   += 1

        obs        = self._flatten_obs(raw_obs)
        reward     = self._compute_reward(raw_obs, raw_reward, action)
        terminated = bool(done and raw_reward > 0)   # genuine task success
        truncated  = bool(self._step_count >= self.horizon and not terminated)

        info["success"]      = float(terminated)
        info["raw_reward"]   = float(raw_reward)
        info["step"]         = self._step_count

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB frame (H, W, 3) uint8, or None for 'human' mode."""
        if self.render_mode == "rgb_array":
            frame = self._env.sim.render(
                camera_name="agentview",
                width=256,
                height=256,
                depth=False,
            )
            return frame[::-1]  # robosuite renders upside-down
        else:
            self._env.render()
            return None

    def close(self) -> None:
        self._env.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers (used by evaluate_policy_robomimic)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def last_raw_obs(self) -> Optional[Dict]:
        """Raw (un-flattened) observation dict from most recent step/reset."""
        return self._last_raw_obs

    def get_obs_keys(self) -> List[str]:
        """Ordered list of observation keys used for flattening."""
        return list(self._obs_keys)