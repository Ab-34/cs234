"""
Robomimic Environment Wrapper

This module provides a wrapper for robomimic/robosuite environments and utilities
to load expert demonstrations from HDF5 datasets (state-based only).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import h5py


class RobomimicEnvWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for robomimic/robosuite environments.
    
    This wrapper:
    - Wraps a robosuite environment used by robomimic
    - Uses state-based observations only (low_dim)
    - Provides a Gymnasium-compatible interface
    - Tracks success for evaluation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        env_name: str = "Lift",
        robots: str = "Panda",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = False,
        use_camera_obs: bool = False,
        control_freq: int = 20,
        horizon: int = 100,
        render_mode: Optional[str] = None,
        full_loss: bool = True,
        **kwargs
    ):
        """
        Args:
            env_name: Name of the robosuite environment (e.g., "Lift", "Can", "Square")
            robots: Robot type (e.g., "Panda", "Sawyer")
            has_renderer: Enable on-screen rendering
            has_offscreen_renderer: Enable off-screen rendering
            use_camera_obs: Use camera observations (we'll use False for state-based)
            control_freq: Control frequency
            horizon: Episode horizon
            render_mode: Rendering mode for Gymnasium compatibility
            **kwargs: Additional arguments for robosuite environment
        """
        super().__init__()
        
        import robosuite as suite
        
        # Store parameters
        self.env_name = env_name
        self.robots = robots
        self.render_mode = render_mode
        self.horizon = horizon
        
        # Create robosuite environment with default controller settings
        # This matches the configuration used to collect the robomimic datasets
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=has_renderer or (render_mode == "human"),
            has_offscreen_renderer=has_offscreen_renderer or (render_mode == "rgb_array"),
            use_camera_obs=use_camera_obs,  # False for state-based
            control_freq=control_freq,
            horizon=horizon,
            # Don't specify controller_configs - use dataset defaults
            **kwargs
        )
        
        # Extract observation and action spaces
        # For state-based: get the concatenated state observation
        obs = self.env.reset()
        
        # Define the state keys that match robomimic dataset format
        # These are the "core" observations used in the datasets
        self.state_keys = [
            'object-state',  # or 'object' in older versions
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'robot0_gripper_qvel',
            'robot0_joint_pos_cos',
            'robot0_joint_pos_sin',
            'robot0_joint_vel',
        ]
        
        # Concatenate state observations to match dataset format
        state_obs = self._extract_state_obs(obs)
        obs_dim = state_obs.shape[0]
        
        # Define observation space (continuous state vector)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space to match the dataset (7 dimensions for Panda)
        # Dataset actions: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # Use 7 dimensions regardless of what the environment reports
        self.dataset_action_dim = 7
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.dataset_action_dim,),
            dtype=np.float32
        )
        
        self.steps = 0

        self.table_height = None
        self.full_loss = full_loss
        print("FULL LOSS:", self.full_loss)


    def compute_dense_reward_from_info_2(self, info: dict) -> float:
        """
        Dense reward encouraging:
        1. Center gripper over cube (XY alignment)
        2. Hover slightly above cube (Z alignment)
        3. Align gripper pointing downward
        4. Close gripper ONLY when centered
        5. Lift cube AFTER proper grasp
        # FULL REWARD FUNCTION OLDER ONE
        """

        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos = np.array(info["robot0_eef_pos"])

        if "cube_pos" in info:
            cube_pos = np.array(info["cube_pos"])
        elif "object" in info:
            cube_pos = np.array(info["object"])
        else:
            return 0.0

        # --------------------------------------------------
        # 1. XY Centering
        # --------------------------------------------------
        xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
        r_xy = -xy_dist

        # --------------------------------------------------
        # 2. Z Hover Alignment
        # --------------------------------------------------
        target_hover_height = 0.06  # 2 cm above cube
        z_error = abs(eef_pos[2] - (cube_pos[2] + target_hover_height))
        r_z = -z_error

        # --------------------------------------------------
        # 3. Orientation (gripper pointing down)
        # --------------------------------------------------
        r_orientation = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = np.array(info["robot0_eef_quat"])

            # Compute gripper z-axis in world frame
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx*qx + qy*qy)
            ])

            # Dot with downward vector
            downward = np.array([0., 0., -1.])
            alignment = np.dot(eef_z_world, downward)

            # Only care about orientation when near cube
            proximity = float(np.exp(-5.0 * xy_dist))
            r_orientation = proximity * alignment

        # --------------------------------------------------
        # 4. Gripper State
        # --------------------------------------------------
        if "robot0_gripper_qpos" in info:
            gripper_qpos = np.array(info["robot0_gripper_qpos"])
            # gripper_closed = gripper_qpos[0] > 0.5
            # gripper_open = gripper_qpos[0] < 0.1
            gripper_closed = gripper_qpos[0] < 0.01   # nearly shut
            gripper_open   = gripper_qpos[0] > 0.03   # fully open
        else:
            gripper_closed = False
            gripper_open = True

        # --------------------------------------------------
        # 5. Grasp Logic
        # --------------------------------------------------
        r_grasp = 0.0

        # Encourage open gripper when approaching
        if xy_dist < 0.06 and gripper_open:
            r_grasp += 0.5

        # Reward closing ONLY if centered and aligned in Z
        if xy_dist < 0.03 and z_error < 0.02 and gripper_closed:
            r_grasp += 6.0

        # Penalize closing when not aligned
        if gripper_closed and xy_dist > 0.03:
            r_grasp -= 1.0

        # --------------------------------------------------
        # 6. Lift Reward (only if grasped properly)
        # --------------------------------------------------
        cube_height = cube_pos[2]

        if self.table_height is None:
            self.table_height = cube_height

        height_above_table = cube_height - self.table_height

        r_lift = 0.0

        if gripper_closed and xy_dist < 0.02:
            r_lift = 10.0 * max(0.0, height_above_table)

        # Bonus for clear lift
        if height_above_table > 0.04 and gripper_closed:
            r_lift += 5.0

        # --------------------------------------------------
        # Final Weighted Sum
        # --------------------------------------------------
        w_xy = 3.0
        w_z = 2.0
        w_orient = 1.0
        w_grasp = 2.0
        w_lift = 10.0

        dense_reward = (
            w_xy * r_xy +
            w_z * r_z +
            w_orient * r_orientation +
            w_grasp * r_grasp +
            w_lift * r_lift
        )

        return float(dense_reward)
    

    def compute_dense_reward_from_info_not_working_to_lower(self, info: dict) -> float:
        """
        Staged reward function with gating:
        Phase 1 — Move EEF above cube (XY + Z hover)
        Phase 2 — Orient gripper downward
        Phase 3 — Close gripper (only when positioned)
        Phase 4 — Lift (only when grasped)

        Each phase's reward is always active but scaled by
        how well the prior phases are satisfied, creating a
        natural curriculum without hard if/else branches.



        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        # ---------- CURRICULUM / PROGRESSIVE HOVER ----------
        # Slowly decrease hover from old learned height (0.1) toward target (0.01)
        if not hasattr(self, "training_step"):
            self.training_step = 0  # initialize step counter

        START_HOVER = 0.1
        END_HOVER   = 0.01
        PROGRESS_RATE = 1e-6  # adjust for speed of lowering
        HOVER_HEIGHT = max(END_HOVER, START_HOVER - self.training_step * PROGRESS_RATE)

        z_target = cube_pos[2] + HOVER_HEIGHT
        z_error  = float(abs(eef_pos[2] - z_target))

        print("Z ERROR IS", z_error, " HOVER HEIGHT IS ", z_target, " Z VALUE IS, ", eef_pos[2], " TO REDUCE IS ", self.training_step * PROGRESS_RATE)

        # Smooth exponential shaping — always positive, peak = 0
        r_xy = float(np.exp(-8.0  * xy_dist))
        r_z  = float(np.exp(-150.0 * z_error))

        approach_score = r_xy * r_z

        # ---------- LOWER HEIGHT BONUS ----------
        # Reward agent for going below old hover
        LOWER_BONUS = max(0.0, START_HOVER - (eef_pos[2] - cube_pos[2]))
        r_z += 50.0 * LOWER_BONUS  # scale bonus as needed

        # ---------- HIGH Z PENALTY ----------
        # Penalize staying above old hover (discourage 0.1 plateau)
        if eef_pos[2] - cube_pos[2] > START_HOVER - 0.01:  # slightly below 0.1
            r_z *= 0.01  # scale down reward

        # ── Phase 2: Orientation ─────────────────────────────────────────────
        r_orient = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = info["robot0_eef_quat"]
            # Gripper local -Z axis in world frame
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx**2 + qy**2)
            ])
            alignment = float(np.dot(eef_z_world, [0., 0., -1.]))  # -1..1
            r_orient = (alignment + 1.0) / 2.0      # remap to 0..1

        # ── Phase 3: Gripper Close ────────────────────────────────────────────
        r_grasp = 0.0
        gripper_closed = False
        if "robot0_gripper_qpos" in info:
            g = float(np.array(info["robot0_gripper_qpos"]).mean())
            # Normalize: 0 = fully open, 1 = fully closed
            # Adjust these bounds to match your robot's actual qpos range
            OPEN_VAL, CLOSE_VAL = 0.04, 0.0
            gripper_norm = float(np.clip(
                (OPEN_VAL - g) / (OPEN_VAL - CLOSE_VAL + 1e-6), 0.0, 1.0
            ))
            gripper_closed = gripper_norm > 0.7

            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            r_grasp   = readiness * gripper_norm     # shaped, not binary

        # ── Phase 4: Lift ─────────────────────────────────────────────────────
        r_lift = 0.0
        if self.table_height is None:
            # Set once at the very first step (cube should be resting on table)
            self.table_height = float(cube_pos[2])

        height_above_table = float(cube_pos[2]) - self.table_height

        if gripper_closed and xy_dist < 0.05:
            # Shaped lift: reward proportional to height, with a bonus plateau
            r_lift = float(np.tanh(10.0 * max(0.0, height_above_table)))
            if height_above_table > 0.05:
                r_lift += 1.0                        # clear lift bonus

        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            1.0  * r_orient    +   # orientation always encouraged
            4.0  * r_grasp     +   # grasp gated by approach quality
            10.0 * r_lift          # lift is the main task signal
        )
        self.training_step +=1
        return float(reward)
    
    def compute_dense_reward_from_info_align1cm(self, info: dict) -> float:
        """
        Staged reward function with gating:
        Phase 1 — Move EEF above cube (XY + Z hover)
        Phase 2 — Orient gripper downward
        Phase 3 — Close gripper (only when positioned)
        Phase 4 — Lift (only when grasped)

        Each phase's reward is always active but scaled by
        how well the prior phases are satisfied, creating a
        natural curriculum without hard if/else branches.


        # WORKED PERFECT;Y TO ALIGN AND REACH 1cm ABOVE CUBE

        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        HOVER_HEIGHT = 0.01                        # target height above cube center
        z_target     = cube_pos[2] + HOVER_HEIGHT
        z_error      = float(abs(eef_pos[2] - z_target))

        # print("Z ERROR IS ", z_error)

        # Smooth exponential shaping — always positive, peak = 0
        r_xy = float(np.exp(-8.0  * xy_dist))        # 1.0 when perfect, ~0 at 0.3m
        # r_z  = float(np.exp(-15.0 * z_error))        # tight Z tolerance

        z_above_target = max(0.0, eef_pos[2] - (cube_pos[2] + HOVER_HEIGHT))
        r_z = float(np.exp(-15.0 * z_above_target))

        approach_score = r_xy * r_z                  # both must be good simultaneously

        # ── Phase 1.5: Descend bonus (lures agent from 0.1 → 0.01) ──────────────────
        r_descend = 0.0
        z_above_cube = max(0.0, eef_pos[2] - cube_pos[2])  # 0 when at cube height
        if r_xy > 0.6:  # only activate when XY is already aligned (preserve Phase 1 skill)
            # Peaks when eef is right at cube level, decays as height increases
            r_descend = r_xy * float(np.exp(-20.0 * z_above_cube))

        # ── Phase 2: Orientation ─────────────────────────────────────────────
        r_orient = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = info["robot0_eef_quat"]
            # Gripper local -Z axis in world frame
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx**2 + qy**2)
            ])
            alignment = float(np.dot(eef_z_world, [0., 0., -1.]))  # -1..1
            r_orient = (alignment + 1.0) / 2.0      # remap to 0..1

        # ── Phase 3: Gripper Close ────────────────────────────────────────────
        r_grasp = 0.0
        gripper_closed = False
        if "robot0_gripper_qpos" in info:
            g = float(np.array(info["robot0_gripper_qpos"]).mean())
            # Normalize: 0 = fully open, 1 = fully closed
            # Adjust these bounds to match your robot's actual qpos range
            OPEN_VAL, CLOSE_VAL = 0.04, 0.0
            gripper_norm = float(np.clip(
                (OPEN_VAL - g) / (OPEN_VAL - CLOSE_VAL + 1e-6), 0.0, 1.0
            ))
            gripper_closed = gripper_norm > 0.7

            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            r_grasp   = readiness * gripper_norm     # shaped, not binary

        # ── Phase 4: Lift ─────────────────────────────────────────────────────
        r_lift = 0.0
        if self.table_height is None:
            # Set once at the very first step (cube should be resting on table)
            self.table_height = float(cube_pos[2])

        height_above_table = float(cube_pos[2]) - self.table_height

        if gripper_closed and xy_dist < 0.05:
            # Shaped lift: reward proportional to height, with a bonus plateau
            r_lift = float(np.tanh(10.0 * max(0.0, height_above_table)))
            if height_above_table > 0.05:
                r_lift += 1.0                        # clear lift bonus

        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            6.0  * r_descend   +
            1.0  * r_orient    +   # orientation always encouraged
            4.0  * r_grasp     +   # grasp gated by approach quality
            10.0 * r_lift          # lift is the main task signal
        )

        return float(reward)
    
    def compute_dense_reward_from_info_working_hover_align(self, info: dict) -> float:
        """
        Staged reward function with gating:
        Phase 1 — Move EEF above cube (XY + Z hover)
        Phase 2 — Orient gripper downward
        Phase 3 — Close gripper (only when positioned)
        Phase 4 — Lift (only when grasped)

        Each phase's reward is always active but scaled by
        how well the prior phases are satisfied, creating a
        natural curriculum without hard if/else branches.


        # WORKED PERFECT;Y TO ALIGN AND REACH 10CM ABOVE CUBE

        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        HOVER_HEIGHT = 0.01                        # target height above cube center
        z_target     = cube_pos[2] + HOVER_HEIGHT
        z_error      = float(abs(eef_pos[2] - z_target))

        # Smooth exponential shaping — always positive, peak = 0
        r_xy = float(np.exp(-8.0  * xy_dist))        # 1.0 when perfect, ~0 at 0.3m
        r_z  = float(np.exp(-15.0 * z_error))        # tight Z tolerance

        approach_score = r_xy * r_z                  # both must be good simultaneously

        # ── Phase 2: Orientation ─────────────────────────────────────────────
        r_orient = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = info["robot0_eef_quat"]
            # Gripper local -Z axis in world frame
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx**2 + qy**2)
            ])
            alignment = float(np.dot(eef_z_world, [0., 0., -1.]))  # -1..1
            r_orient = (alignment + 1.0) / 2.0      # remap to 0..1

        # ── Phase 3: Gripper Close ────────────────────────────────────────────
        r_grasp = 0.0
        gripper_closed = False
        if "robot0_gripper_qpos" in info:
            g = float(np.array(info["robot0_gripper_qpos"]).mean())
            # Normalize: 0 = fully open, 1 = fully closed
            # Adjust these bounds to match your robot's actual qpos range
            OPEN_VAL, CLOSE_VAL = 0.04, 0.0
            gripper_norm = float(np.clip(
                (OPEN_VAL - g) / (OPEN_VAL - CLOSE_VAL + 1e-6), 0.0, 1.0
            ))
            gripper_closed = gripper_norm > 0.7

            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            r_grasp   = readiness * gripper_norm     # shaped, not binary

        # ── Phase 4: Lift ─────────────────────────────────────────────────────
        r_lift = 0.0
        if self.table_height is None:
            # Set once at the very first step (cube should be resting on table)
            self.table_height = float(cube_pos[2])

        height_above_table = float(cube_pos[2]) - self.table_height

        if gripper_closed and xy_dist < 0.05:
            # Shaped lift: reward proportional to height, with a bonus plateau
            r_lift = float(np.tanh(10.0 * max(0.0, height_above_table)))
            if height_above_table > 0.05:
                r_lift += 1.0                        # clear lift bonus

        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            1.0  * r_orient    +   # orientation always encouraged
            4.0  * r_grasp     +   # grasp gated by approach quality
            10.0 * r_lift          # lift is the main task signal
        )

        return float(reward)

    def compute_dense_reward_from_info(self, info: dict) -> float:
        """
        Staged reward function with gating:
        Phase 1 — Move EEF above cube (XY + Z hover)
        Phase 2 — Orient gripper downward
        Phase 3 — Close gripper (only when positioned)
        Phase 4 — Lift (only when grasped)

        Each phase's reward is always active but scaled by
        how well the prior phases are satisfied, creating a
        natural curriculum without hard if/else branches.

        # compute_dense_reward_from_info_to_test_later
        # compute_dense_reward_from_info_maybe_working_doesnt_give_good_Results

        #3
        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        HOVER_HEIGHT = 0.025                      # target height above cube center
        z_target     = cube_pos[2] + HOVER_HEIGHT
        z_error      = float(abs(eef_pos[2] - z_target))

        # Smooth exponential shaping — always positive, peak = 0
        r_xy = float(np.exp(-8.0  * xy_dist))        # 1.0 when perfect, ~0 at 0.3m
        r_z  = float(np.exp(-15.0 * z_error))        # tight Z tolerance

        approach_score = r_xy * r_z                  # both must be good simultaneously

        # ── Phase 2: Orientation ─────────────────────────────────────────────
        r_orient = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = info["robot0_eef_quat"]
            # Gripper local -Z axis in world frame
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx**2 + qy**2)
            ])
            alignment = float(np.dot(eef_z_world, [0., 0., -1.]))  # -1..1

            # Angular error in radians (0 = perfect, π = flipped)
            # angle_error = float(np.arccos(np.clip(alignment, -1.0, 1.0)))
            # Sharp exponential: near-zero reward unless nearly vertical
            # r_orient = float(np.exp(-5.0 * angle_error))   # ~1.0 at 0°, ~0.08 at 50°


            r_orient = (alignment + 1.0) / 2.0      # remap to 0..1

        # ── Phase 3: Gripper Close ────────────────────────────────────────────
        r_grasp = 0.0
        gripper_closed = False
        if "robot0_gripper_qpos" in info:
            # g = float(np.array(info["robot0_gripper_qpos"]).mean())
            g = float(np.abs(np.array(info["robot0_gripper_qpos"])).mean())

            # print("GRIPPER POS IS , ", g)

            # Normalize: 0 = fully open, 1 = fully closed
            # Adjust these bounds to match your robot's actual qpos range
            OPEN_VAL, CLOSE_VAL = 0.04, 0.0
            gripper_norm = float(np.clip(
                (OPEN_VAL - g) / (OPEN_VAL - CLOSE_VAL + 1e-6), 0.0, 1.0
            ))
            gripper_closed = gripper_norm > 0.7
            r_proximity_grasp = float(np.exp(-3.0 * xy_dist)) * gripper_norm


            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            # r_grasp = r_proximity_grasp + readiness * gripper_norm

            READINESS_THRESHOLD = 0.6  # tune this
            r_grasp = gripper_norm * (readiness - READINESS_THRESHOLD) / (1.0 - READINESS_THRESHOLD + 1e-6)
            r_grasp = float(np.clip(r_grasp, -1.0, 1.0))

        # ── Phase 4: Lift ─────────────────────────────────────────────────────
        r_lift = 0.0
        if self.table_height is None:
            # Set once at the very first step (cube should be resting on table)
            self.table_height = float(cube_pos[2])

        height_above_table = float(cube_pos[2]) - self.table_height

        # if gripper_closed and xy_dist < 0.05:
        #     # Shaped lift: reward proportional to height, with a bonus plateau
        #     r_lift = float(np.tanh(10.0 * max(0.0, height_above_table)))
        #     if height_above_table > 0.05:
        #         r_lift += 1.0                        # clear lift bonus


        # lift_ready = gripper_norm * r_orient * r_xy  # i.e., ready when gripping, oriented, close
        # r_lift = lift_ready * float(np.tanh(10.0 * max(0.0, height_above_table)))
        # if height_above_table > 0.05:
        #     r_lift += 1.0

        # Instead of a strict product gate, use a softer gate:
        grasp_quality = float(np.clip(gripper_norm * r_xy, 0.0, 1.0))  # simpler 2-factor gate

        lift_height_reward = float(np.tanh(10.0 * max(0.0, height_above_table)))
        r_lift = grasp_quality * lift_height_reward

        # Keep the plateau bonus, but also add a *shaped* pre-lift bonus
        # so the robot is incentivized to explore upward movement:
        if height_above_table > 0.02:   # lower threshold to start rewarding sooner
            r_lift += 0.5
        if height_above_table > 0.05:
            r_lift += 1.0               # full bonus for clear lift



        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            10.0  * r_orient    +   # orientation always encouraged
            6.0  * r_grasp     +   # grasp gated by approach quality
            5000.0 * r_lift          # lift is the main task signal
        )

        return float(reward)
    

    def compute_dense_reward_from_info_not_good_latest(self, info: dict) -> float:
        """
        Staged reward function with gating:
        Phase 1 — Move EEF above cube (XY + Z hover)
        Phase 2 — Orient gripper downward
        Phase 3 — Close gripper (only when positioned)
        Phase 4 — Lift (only when grasped)

        Each phase's reward is always active but scaled by
        how well the prior phases are satisfied, creating a
        natural curriculum without hard if/else branches.

        # TESTING WITH BETTER ALIGNMENT AND PICKING UP

        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        HOVER_HEIGHT = 0.02                      # target height above cube center
        z_target     = cube_pos[2] + HOVER_HEIGHT
        z_error      = float(abs(eef_pos[2] - z_target))

        # Smooth exponential shaping — always positive, peak = 0
        r_xy = float(np.exp(-8.0  * xy_dist))        # 1.0 when perfect, ~0 at 0.3m
        r_z  = float(np.exp(-15.0 * z_error))        # tight Z tolerance

        approach_score = r_xy * r_z                  # both must be good simultaneously

        # ── Phase 2: Orientation (vertical approach + roll alignment) ────────────
        r_orient = 0.0
        if "robot0_eef_quat" in info:
            qx, qy, qz, qw = info["robot0_eef_quat"]

            # ── Term 1: Gripper Z-axis points down ──────────────────────────────
            eef_z_world = np.array([
                2*(qx*qz + qy*qw),
                2*(qy*qz - qx*qw),
                1 - 2*(qx**2 + qy**2)
            ])
            alignment_z = float(np.dot(eef_z_world, [0., 0., -1.]))
            angle_error_z = float(np.arccos(np.clip(alignment_z, -1.0, 1.0)))
            r_vertical = float(np.exp(-5.0 * angle_error_z))

            # ── Term 2: Gripper X-axis aligns with block's long axis ────────────
            # Gripper local X-axis in world frame
            eef_x_world = np.array([
                1 - 2*(qy**2 + qz**2),
                2*(qx*qy + qz*qw),
                2*(qx*qz - qy*qw)
            ])

            # Get the block's facing direction from its quaternion
            # If the block has no orientation info, fall back to world X axis
            if "cube_quat" in info:
                bx, by, bz, bw = info["cube_quat"]
                block_x_world = np.array([
                    1 - 2*(by**2 + bz**2),
                    2*(bx*by + bz*bw),
                    2*(bx*bz - by*bw)
                ])
            else:
                block_x_world = np.array([1., 0., 0.])  # assume axis-aligned block

            # Use abs() because gripper can grip from either side (180° symmetry)
            alignment_roll = float(abs(np.dot(eef_x_world, block_x_world)))
            # alignment_roll: 1.0 = perfectly aligned, 0.0 = 90° off
            angle_error_roll = float(np.arccos(np.clip(alignment_roll, 0.0, 1.0)))
            r_roll = float(np.exp(-5.0 * angle_error_roll))

            # Combine: both must be satisfied
            r_orient = r_vertical * r_roll

        # ── Phase 3: Gripper Close ────────────────────────────────────────────
        r_grasp = 0.0
        gripper_closed = False
        if "robot0_gripper_qpos" in info:
            # g = float(np.array(info["robot0_gripper_qpos"]).mean())
            g = float(np.abs(np.array(info["robot0_gripper_qpos"])).mean())

            # print("GRIPPER POS IS , ", g)

            # Normalize: 0 = fully open, 1 = fully closed
            # Adjust these bounds to match your robot's actual qpos range
            OPEN_VAL, CLOSE_VAL = 0.04, 0.0
            gripper_norm = float(np.clip(
                (OPEN_VAL - g) / (OPEN_VAL - CLOSE_VAL + 1e-6), 0.0, 1.0
            ))
            gripper_closed = gripper_norm > 0.7
            r_proximity_grasp = float(np.exp(-3.0 * xy_dist)) * gripper_norm


            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            # r_grasp = r_proximity_grasp + readiness * gripper_norm

            READINESS_THRESHOLD = 0.6  # tune this
            r_grasp = gripper_norm * (readiness - READINESS_THRESHOLD) / (1.0 - READINESS_THRESHOLD + 1e-6)
            r_grasp = float(np.clip(r_grasp, -1.0, 1.0)) + r_proximity_grasp 

        # ── Phase 4: Lift ─────────────────────────────────────────────────────
        r_lift = 0.0
        if self.table_height is None:
            # Set once at the very first step (cube should be resting on table)
            self.table_height = float(cube_pos[2])

        height_above_table = float(cube_pos[2]) - self.table_height

        # if gripper_closed and xy_dist < 0.05:
        #     # Shaped lift: reward proportional to height, with a bonus plateau
        #     r_lift = float(np.tanh(10.0 * max(0.0, height_above_table)))
        #     if height_above_table > 0.05:
        #         r_lift += 1.0                        # clear lift bonus


        # lift_ready = gripper_norm * r_orient * r_xy  # i.e., ready when gripping, oriented, close
        # r_lift = lift_ready * float(np.tanh(10.0 * max(0.0, height_above_table)))
        # if height_above_table > 0.05:
        #     r_lift += 1.0

        # Instead of a strict product gate, use a softer gate:
        grasp_quality = float(np.clip(gripper_norm * r_xy, 0.0, 1.0))  # simpler 2-factor gate

        lift_height_reward = float(np.tanh(10.0 * max(0.0, height_above_table)))
        r_lift = grasp_quality * lift_height_reward

        # Keep the plateau bonus, but also add a *shaped* pre-lift bonus
        # so the robot is incentivized to explore upward movement:
        if height_above_table > 0.02:   # lower threshold to start rewarding sooner
            r_lift += 0.5
        if height_above_table > 0.05:
            r_lift += 1.0               # full bonus for clear lift



        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            2.0  * r_orient    +   # orientation always encouraged
            6.0  * r_grasp     +   # grasp gated by approach quality
            50.0 * r_lift          # lift is the main task signal
        )

        return float(reward)
        
    
    def compute_dense_reward_from_info_11(self, info: dict) -> float:
        """
        Staged curriculum reward for pick-and-place.

        Core fix vs prior version:
        The robot was learning to close its gripper immediately and ram the block.
        Root cause: r_grasp = gripper_norm * r_proximity rewarded closing at ANY
        time the robot was near the cube, even while approaching.

        Solution (three-part fix):
        1. Reward open-gripper approach: robot is explicitly incentivised to arrive
            above the cube with the gripper OPEN.
        2. Steep approach gate on grasping: closing only pays off when approach_score
            is already high (approach_score ** 3 makes the gate very sharp).
        3. Premature-close penalty: explicit negative reward for closing the gripper
            while the approach quality is still low — directly discourages ramming.


        #key = 11
        #FULL COMPLEX REWARD WITH EVERYTHING:

        """
        try:
            # ── Input validation ──────────────────────────────────────────────────
            if "robot0_eef_pos" not in info:
                return 0.0

            eef_pos = np.asarray(info["robot0_eef_pos"], dtype=np.float64).ravel()
            if eef_pos.shape[0] < 3 or not np.all(np.isfinite(eef_pos)):
                return 0.0

            cube_raw = info.get("cube_pos", info.get("object", None))
            if cube_raw is None:
                return 0.0
            cube_pos = np.asarray(cube_raw, dtype=np.float64).ravel()
            if cube_pos.shape[0] < 3 or not np.all(np.isfinite(cube_pos)):
                return 0.0

            # ── Phase 1 — XY Alignment ────────────────────────────────────────────
            xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))
            r_xy    = float(np.exp(-8.0 * xy_dist))

            # ── Phase 2 — Z Hover ─────────────────────────────────────────────────
            HOVER_HEIGHT = 0.03                          # 3 cm above cube centre
            z_target     = float(cube_pos[2]) + HOVER_HEIGHT
            z_error      = abs(float(eef_pos[2]) - z_target)
            r_z          = float(np.exp(-12.0 * z_error))

            # Combined approach quality (0..1); used as the gating signal below
            approach_score = r_xy * r_z

            # ── Phase 3 — Orientation ─────────────────────────────────────────────
            r_orient   = 0.0
            r_vertical = 0.0
            r_roll     = 0.0

            if "robot0_eef_quat" in info:
                try:
                    quat = np.asarray(info["robot0_eef_quat"], dtype=np.float64).ravel()
                    if quat.shape[0] == 4 and np.all(np.isfinite(quat)):
                        q_norm = float(np.linalg.norm(quat))
                        if q_norm < 1e-6:
                            raise ValueError("zero quaternion")
                        qx, qy, qz, qw = quat / q_norm

                        eef_z = np.array([
                            2.0 * (qx * qz + qy * qw),
                            2.0 * (qy * qz - qx * qw),
                            1.0 - 2.0 * (qx ** 2 + qy ** 2),
                        ])
                        eef_z /= max(float(np.linalg.norm(eef_z)), 1e-6)
                        dot_z       = float(np.clip(np.dot(eef_z, [0.0, 0.0, -1.0]), -1.0, 1.0))
                        r_vertical  = float(np.exp(-5.0 * np.arccos(dot_z)))

                        eef_x = np.array([
                            1.0 - 2.0 * (qy ** 2 + qz ** 2),
                            2.0 * (qx * qy + qz * qw),
                            2.0 * (qx * qz - qy * qw),
                        ])
                        eef_x /= max(float(np.linalg.norm(eef_x)), 1e-6)

                        block_x = np.array([1.0, 0.0, 0.0])
                        if "cube_quat" in info:
                            try:
                                bq = np.asarray(info["cube_quat"], dtype=np.float64).ravel()
                                if bq.shape[0] == 4 and np.all(np.isfinite(bq)):
                                    bn = float(np.linalg.norm(bq))
                                    if bn > 1e-6:
                                        bx, by, bz, bw = bq / bn
                                        cand = np.array([
                                            1.0 - 2.0 * (by ** 2 + bz ** 2),
                                            2.0 * (bx * by + bz * bw),
                                            2.0 * (bx * bz - by * bw),
                                        ])
                                        cn = float(np.linalg.norm(cand))
                                        if cn > 1e-6:
                                            block_x = cand / cn
                            except Exception:
                                pass

                        dot_roll = float(np.clip(abs(np.dot(eef_x, block_x)), 0.0, 1.0))
                        r_roll   = float(np.exp(-5.0 * np.arccos(dot_roll)))

                        r_orient = r_vertical * r_roll

                except Exception:
                    r_orient = 0.0

            # ── Phase 4 — Gripper (open-approach + gated close) ───────────────────
            r_approach_open    = 0.0   # reward arriving open — the key anti-ramming term
            r_grasp            = 0.0   # reward closing when well positioned
            r_premature_close  = 0.0   # penalty for closing while not yet positioned
            gripper_norm       = 0.0

            if "robot0_gripper_qpos" in info:
                try:
                    gqpos = np.asarray(info["robot0_gripper_qpos"], dtype=np.float64).ravel()
                    if gqpos.size > 0 and np.all(np.isfinite(gqpos)):
                        g = float(np.abs(gqpos).mean())

                        OPEN_VAL  = 0.04
                        CLOSE_VAL = 0.00
                        denom     = abs(OPEN_VAL - CLOSE_VAL) + 1e-6
                        gripper_norm = float(np.clip((OPEN_VAL - g) / denom, 0.0, 1.0))
                        gripper_open = 1.0 - gripper_norm        # 1 = fully open

                        # ── Term A: Open-gripper approach ─────────────────────────
                        # Rewards the robot for being above the cube WITH the gripper
                        # open. This creates a strong curriculum: first learn to
                        # navigate to the hover position while staying open.
                        # approach_score is high only when BOTH r_xy and r_z are good.
                        r_approach_open = approach_score * gripper_open

                        # ── Term B: Gated grasp reward ────────────────────────────
                        # approach_score ** 3 creates a very steep gate:
                        #   score=0.5  → gate=0.125  (almost no reward for closing)
                        #   score=0.8  → gate=0.512  (moderate reward)
                        #   score=0.95 → gate=0.857  (strong reward — robot is ready)
                        # This means the robot must FULLY solve approach before closing
                        # pays off more than the open-approach reward above.
                        approach_gate   = approach_score ** 3
                        orient_bonus    = 0.4 + 0.6 * r_orient   # 0.4..1.0 multiplier
                        r_grasp         = approach_gate * gripper_norm * orient_bonus
                        r_grasp         = float(np.clip(r_grasp, 0.0, 1.0))

                        # ── Term C: Premature-close penalty ───────────────────────
                        # When approach_score is low but gripper is closing, penalise.
                        # (1 - approach_score) is high when far/misaligned.
                        # gripper_norm is high when closing.
                        # The product gives max penalty exactly when ramming occurs.
                        bad_close_score    = (1.0 - approach_score) * gripper_norm
                        r_premature_close  = float(np.clip(bad_close_score, 0.0, 1.0))

                except Exception:
                    gripper_norm      = 0.0
                    r_grasp           = 0.0
                    r_approach_open   = 0.0
                    r_premature_close = 0.0

            # ── Phase 5 — Lift ────────────────────────────────────────────────────
            r_lift = 0.0
            try:
                if self.table_height is None:
                    self.table_height = float(cube_pos[2])

                height_above_table = max(0.0, float(cube_pos[2]) - float(self.table_height))

                # Gate: robot must be gripping AND centred XY to earn lift reward.
                # Orientation is excluded from the gate — once grasped, adding
                # orientation as a gate would block the lift signal entirely.
                grasp_gate    = float(np.clip(gripper_norm * r_xy, 0.0, 1.0))

                # Smooth continuous height reward (active from the very first mm)
                r_lift_shaped = float(np.tanh(15.0 * height_above_table))
                r_lift        = grasp_gate * r_lift_shaped

                # Progressive plateau bonuses, always scaled by grasp_gate
                if height_above_table > 0.02:
                    r_lift += 0.5 * grasp_gate
                if height_above_table > 0.05:
                    r_lift += 1.0 * grasp_gate

            except Exception:
                r_lift = 0.0

            # ── Weighted combination ──────────────────────────────────────────────
            #
            # Weight rationale:
            #   r_xy / r_z          low-moderate: always guide position, don't dominate
            #   r_orient            moderate: always useful but position matters more
            #   r_approach_open     KEY NEW TERM: teaches "arrive open" curriculum
            #   r_premature_close   NEGATIVE: directly punishes ramming behaviour
            #   r_grasp             strong: only earns out when truly positioned
            #   r_lift              dominant: the actual task objective
            #
            reward = (
                3.0  * r_xy             +
                2.0  * r_z              +
                10.0  * r_orient         +
                10.0  * r_approach_open  +   # ← NEW: incentivise arriving open
            -10.0  * r_premature_close+   # ← NEW: punish ramming/premature close
                10.0 * r_grasp          +
                50000.0 * r_lift
            )

            return reward

        except Exception:
            return 0.0
        

    def compute_dense_reward_from_info_basically_just_height(self, info: dict) -> float:
        """
        Staged curriculum reward for pick-and-place.

        Core fix vs prior version:
        The robot was learning to close its gripper immediately and ram the block.
        Root cause: r_grasp = gripper_norm * r_proximity rewarded closing at ANY
        time the robot was near the cube, even while approaching.

        Solution (three-part fix):
        1. Reward open-gripper approach: robot is explicitly incentivised to arrive
            above the cube with the gripper OPEN.
        2. Steep approach gate on grasping: closing only pays off when approach_score
            is already high (approach_score ** 3 makes the gate very sharp).
        3. Premature-close penalty: explicit negative reward for closing the gripper
            while the approach quality is still low — directly discourages ramming.
        """
        try:
            # ── Input validation ──────────────────────────────────────────────────
            if "robot0_eef_pos" not in info:
                return 0.0

            eef_pos = np.asarray(info["robot0_eef_pos"], dtype=np.float64).ravel()
            if eef_pos.shape[0] < 3 or not np.all(np.isfinite(eef_pos)):
                return 0.0

            cube_raw = info.get("cube_pos", info.get("object", None))
            if cube_raw is None:
                return 0.0
            cube_pos = np.asarray(cube_raw, dtype=np.float64).ravel()
            if cube_pos.shape[0] < 3 or not np.all(np.isfinite(cube_pos)):
                return 0.0

            # ── Phase 1 — XY Alignment ────────────────────────────────────────────
            xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))
            r_xy    = float(np.exp(-8.0 * xy_dist))

            # ── Phase 2 — Z Hover ─────────────────────────────────────────────────
            HOVER_HEIGHT = 0.03                          # 3 cm above cube centre
            z_target     = float(cube_pos[2]) + HOVER_HEIGHT
            z_error      = abs(float(eef_pos[2]) - z_target)
            r_z          = float(np.exp(-12.0 * z_error))

            # Combined approach quality (0..1); used as the gating signal below
            approach_score = r_xy * r_z

            # ── Phase 3 — Orientation ─────────────────────────────────────────────
            r_orient   = 0.0
            r_vertical = 0.0
            r_roll     = 0.0

            if "robot0_eef_quat" in info:
                try:
                    quat = np.asarray(info["robot0_eef_quat"], dtype=np.float64).ravel()
                    if quat.shape[0] == 4 and np.all(np.isfinite(quat)):
                        q_norm = float(np.linalg.norm(quat))
                        if q_norm < 1e-6:
                            raise ValueError("zero quaternion")
                        qx, qy, qz, qw = quat / q_norm

                        eef_z = np.array([
                            2.0 * (qx * qz + qy * qw),
                            2.0 * (qy * qz - qx * qw),
                            1.0 - 2.0 * (qx ** 2 + qy ** 2),
                        ])
                        eef_z /= max(float(np.linalg.norm(eef_z)), 1e-6)
                        dot_z       = float(np.clip(np.dot(eef_z, [0.0, 0.0, -1.0]), -1.0, 1.0))
                        r_vertical  = float(np.exp(-5.0 * np.arccos(dot_z)))

                        eef_x = np.array([
                            1.0 - 2.0 * (qy ** 2 + qz ** 2),
                            2.0 * (qx * qy + qz * qw),
                            2.0 * (qx * qz - qy * qw),
                        ])
                        eef_x /= max(float(np.linalg.norm(eef_x)), 1e-6)

                        block_x = np.array([1.0, 0.0, 0.0])
                        if "cube_quat" in info:
                            try:
                                bq = np.asarray(info["cube_quat"], dtype=np.float64).ravel()
                                if bq.shape[0] == 4 and np.all(np.isfinite(bq)):
                                    bn = float(np.linalg.norm(bq))
                                    if bn > 1e-6:
                                        bx, by, bz, bw = bq / bn
                                        cand = np.array([
                                            1.0 - 2.0 * (by ** 2 + bz ** 2),
                                            2.0 * (bx * by + bz * bw),
                                            2.0 * (bx * bz - by * bw),
                                        ])
                                        cn = float(np.linalg.norm(cand))
                                        if cn > 1e-6:
                                            block_x = cand / cn
                            except Exception:
                                pass

                        dot_roll = float(np.clip(abs(np.dot(eef_x, block_x)), 0.0, 1.0))
                        r_roll   = float(np.exp(-5.0 * np.arccos(dot_roll)))

                        r_orient = r_vertical * r_roll

                except Exception:
                    r_orient = 0.0

            # ── Phase 4 — Gripper (open-approach + gated close) ───────────────────
            r_approach_open    = 0.0   # reward arriving open — the key anti-ramming term
            r_grasp            = 0.0   # reward closing when well positioned
            r_premature_close  = 0.0   # penalty for closing while not yet positioned
            gripper_norm       = 0.0

            if "robot0_gripper_qpos" in info:
                try:
                    gqpos = np.asarray(info["robot0_gripper_qpos"], dtype=np.float64).ravel()
                    if gqpos.size > 0 and np.all(np.isfinite(gqpos)):
                        g = float(np.abs(gqpos).mean())

                        OPEN_VAL  = 0.04
                        CLOSE_VAL = 0.00
                        denom     = abs(OPEN_VAL - CLOSE_VAL) + 1e-6
                        gripper_norm = float(np.clip((OPEN_VAL - g) / denom, 0.0, 1.0))
                        gripper_open = 1.0 - gripper_norm        # 1 = fully open

                        # ── Term A: Open-gripper approach ─────────────────────────
                        # Rewards the robot for being above the cube WITH the gripper
                        # open. This creates a strong curriculum: first learn to
                        # navigate to the hover position while staying open.
                        # approach_score is high only when BOTH r_xy and r_z are good.
                        r_approach_open = approach_score * gripper_open

                        # ── Term B: Gated grasp reward ────────────────────────────
                        # approach_score ** 3 creates a very steep gate:
                        #   score=0.5  → gate=0.125  (almost no reward for closing)
                        #   score=0.8  → gate=0.512  (moderate reward)
                        #   score=0.95 → gate=0.857  (strong reward — robot is ready)
                        # This means the robot must FULLY solve approach before closing
                        # pays off more than the open-approach reward above.
                        approach_gate   = approach_score ** 3
                        orient_bonus    = 0.4 + 0.6 * r_orient   # 0.4..1.0 multiplier
                        r_grasp         = approach_gate * gripper_norm * orient_bonus
                        r_grasp         = float(np.clip(r_grasp, 0.0, 1.0))

                        # ── Term C: Premature-close penalty ───────────────────────
                        # When approach_score is low but gripper is closing, penalise.
                        # (1 - approach_score) is high when far/misaligned.
                        # gripper_norm is high when closing.
                        # The product gives max penalty exactly when ramming occurs.
                        bad_close_score    = (1.0 - approach_score) * gripper_norm
                        r_premature_close  = float(np.clip(bad_close_score, 0.0, 1.0))

                except Exception:
                    gripper_norm      = 0.0
                    r_grasp           = 0.0
                    r_approach_open   = 0.0
                    r_premature_close = 0.0

            # ── Phase 5 — Lift ────────────────────────────────────────────────────
            r_lift = 0.0
            try:
                if self.table_height is None:
                    self.table_height = float(cube_pos[2])

                height_above_table = max(0.0, float(cube_pos[2]) - float(self.table_height))

                # Gate: robot must be gripping AND centred XY to earn lift reward.
                # Orientation is excluded from the gate — once grasped, adding
                # orientation as a gate would block the lift signal entirely.
                grasp_gate    = float(np.clip(gripper_norm * r_xy, 0.0, 1.0))

                # Smooth continuous height reward (active from the very first mm)
                r_lift_shaped = float(np.tanh(15.0 * height_above_table))
                r_lift        = grasp_gate * r_lift_shaped

                # Progressive plateau bonuses, always scaled by grasp_gate
                if height_above_table > 0.02:
                    r_lift += 0.5 * grasp_gate
                if height_above_table > 0.05:
                    r_lift += 1.0 * grasp_gate

            except Exception:
                r_lift = 0.0

            # ── Weighted combination ──────────────────────────────────────────────
            #
            # Weight rationale:
            #   r_xy / r_z          low-moderate: always guide position, don't dominate
            #   r_orient            moderate: always useful but position matters more
            #   r_approach_open     KEY NEW TERM: teaches "arrive open" curriculum
            #   r_premature_close   NEGATIVE: directly punishes ramming behaviour
            #   r_grasp             strong: only earns out when truly positioned
            #   r_lift              dominant: the actual task objective
            #
            reward = (
                0.0  * r_xy             +
                0.0  * r_z              +
                0.0  * r_orient         +
                0.0  * r_approach_open  +   # ← NEW: incentivise arriving open
            -0.0  * r_premature_close+   # ← NEW: punish ramming/premature close
                0.0 * r_grasp          +
                50.0 * r_lift
            )

            return float(np.clip(reward, -20.0, 200.0))

        except Exception:
            return 0.0
    
    def get_state(self) -> np.ndarray:
        """Get the current state of the environment."""
        return self.env.sim.get_state()

    def _extract_state_obs(self, obs_dict: Dict) -> np.ndarray:
        """
        Extract and concatenate state observations from observation dictionary.
        
        This matches the observation format used in robomimic datasets.
        We only use the core robot and object observations, excluding derived keys.
        
        Args:
            obs_dict: Dictionary of observations from robosuite
            
        Returns:
            state_obs: Concatenated state observation vector
        """
        state_components = []
        
        # Use only the keys that appear in robomimic datasets
        # These are the "core" observations used for training
        # Order matters! This should match the dataset order
        dataset_keys = [
            'object-state',  # or 'object' in older versions
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'robot0_gripper_qvel',
            'robot0_joint_pos_cos',
            'robot0_joint_pos_sin',
            'robot0_joint_vel',
        ]
        
        for key in dataset_keys:
            if key in obs_dict:
                component = obs_dict[key]
                if isinstance(component, np.ndarray):
                    state_components.append(component.flatten())
                else:
                    state_components.append(np.array([component]))
        
        return np.concatenate(state_components).astype(np.float32)
    
    
    def set_state(self, state):
        """
        Set the simulator state and reset episode tracking.
        Returns the observation from the restored state.
        """
        self.env.sim.reset()
        self.env.sim.set_state(state)
        self.steps = 0
        
        # Get observation from the restored state
        obs_dict = self.env._get_observations()
        return self._extract_state_obs(obs_dict)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        print(f"Reset seed: {seed}")
        
        # Reset robosuite environment
        if state is None:
            # Properly seed all random number generators
            if seed is not None:
                # Seed numpy's global random state (used by robosuite internally)
                np.random.seed(seed)
                # Create RNG for the environment
                rng = np.random.RandomState(seed)
                self.env.rng = rng
            
            # Set deterministic mode
            self.env.deterministic = True
            obs_dict = self.env.reset()
        else:
            self.env.sim.set_state(state)
            obs_dict = self.env.get_obs()
        
        # Extract state observation
        obs = self._extract_state_obs(obs_dict)
        
        self.steps = 0
        
        info = {
            "success": False,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        action = np.array(action, dtype=np.float32)
        
        # Take step in robosuite environment
        obs_dict, reward, done, info = self.env.step(action)
        
        # Extract state observation
        obs = self._extract_state_obs(obs_dict)
        
        self.steps += 1
        
        # robosuite uses "done" flag, we split into terminated/truncated for Gymnasium
        # Check if task succeeded
        success = reward == 1

        reward =  self.compute_dense_reward_from_info(obs_dict) + success*50000  #co
            
        terminated = success  # Task completed successfully
        truncated = done and not success  # Episode ended without success
        
        # Update info
        if not isinstance(info, dict):
            info = {}
        info["success"] = success
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.env.render()
        elif self.render_mode == "rgb_array":
            # Return RGB array for recording
            return self.env.sim.render(
                width=640,
                height=480,
                camera_name="agentview"
            )[::-1]  # Flip vertically
    
    def close(self):
        """Close the environment."""
        self.env.close()


def load_robomimic_trajectories(
    dataset_path: str,
    num_trajectories: Optional[int] = None,
    filter_key: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load expert trajectories from a robomimic HDF5 dataset (state-based).
    
    Args:
        dataset_path: Path to the HDF5 dataset file (e.g., "datasets/low_dim_v141.hdf5")
        num_trajectories: Number of trajectories to load (None = load all)
        filter_key: Optional filter key (e.g., "50_demos" for subset)
        verbose: Print loading information
        
    Returns:
        trajectories: Dictionary containing:
            - observations: List of observation arrays [T, obs_dim]
            - actions: List of action arrays [T, action_dim]
            - rewards: List of reward arrays [T]
            - dones: List of done flags [T]
            - lengths: List of trajectory lengths
            - success: List of success flags (if available)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    if verbose:
        print(f"Loading trajectories from: {dataset_path}")
    
    trajectories = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'lengths': [],
        'success': []
    }
    
    with h5py.File(dataset_path, 'r') as f:
        # Get list of demonstration keys
        if filter_key is not None and filter_key in f['mask']:
            # Use filtered subset
            demo_keys = [f'demo_{i}' for i in np.where(f[f'mask/{filter_key}'][:])[0]]
        else:
            # Use all demonstrations
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        
        # Limit number of trajectories if specified
        if num_trajectories is not None:
            demo_keys = demo_keys[:num_trajectories]
        
        if verbose:
            print(f"Loading {len(demo_keys)} trajectories...")
        
        for demo_key in demo_keys:
            demo = f[f'data/{demo_key}']
            
            # Load observations (state-based)
            # Extract only the observations that match what the environment provides
            if 'obs' in demo:
                obs_dict = demo['obs']
                obs_components = []
                
                # These keys must match the environment's observation keys
                # Order matters - must match _extract_state_obs() in RobomimicEnvWrapper
                dataset_obs_keys = [
                    'object',  # In dataset, this is 'object', in env it's 'object-state'
                    'robot0_eef_pos',
                    'robot0_eef_quat',
                    'robot0_gripper_qpos',
                    'robot0_gripper_qvel',
                    'robot0_joint_pos_cos',
                    'robot0_joint_pos_sin',
                    'robot0_joint_vel',
                ]
                
                for key in dataset_obs_keys:
                    if key in obs_dict:
                        obs_components.append(obs_dict[key][:])
                
                # Concatenate along last dimension to get [T, obs_dim]
                obs = np.concatenate(obs_components, axis=-1)
            else:
                raise ValueError(f"Could not find 'obs' group in {demo_key}")
            
            # Load actions
            actions = demo['actions'][:]
            
            # Load rewards (if available)
            rewards = demo['rewards'][:]
            
            # Load dones (if available)
            dones = demo['dones'][:]
            
            # Get trajectory length
            length = len(actions)
            
            # Check if trajectory was successful
            success = bool((np.array(demo['rewards']) == 1 ).any())
            
            # Append to trajectories
            trajectories['observations'].append(obs.astype(np.float32))
            trajectories['actions'].append(actions.astype(np.float32))
            trajectories['rewards'].append(rewards.astype(np.float32))
            trajectories['dones'].append(dones)
            trajectories['lengths'].append(length)
            trajectories['success'].append(success)
    
    if verbose:
        print(f"Successfully loaded {len(trajectories['observations'])} trajectories")
        print(f"Observation dimension: {trajectories['observations'][0].shape[-1]}")
        print(f"Action dimension: {trajectories['actions'][0].shape[-1]}")
        print(f"Average trajectory length: {np.mean(trajectories['lengths']):.1f}")
        success_rate = np.mean(trajectories['success']) if trajectories['success'] else 0.0
        print(f"Success rate: {success_rate:.2%}")
    
    return trajectories


def get_env_info_from_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Extract environment information from a robomimic dataset.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        
    Returns:
        info: Dictionary containing environment metadata
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    with h5py.File(dataset_path, 'r') as f:
        
        import json
        env_args_raw = f['data'].attrs['env_args']
        # Convert bytes/str array to dict
        if isinstance(env_args_raw, (bytes, str)):
            env_args_str = env_args_raw.decode() if isinstance(env_args_raw, bytes) else env_args_raw
            info = json.loads(env_args_str)
        else:
            info = dict(env_args_raw)
        
        first_demo = f[f'data/demo_0']
        
        # Extract observation dimension using same logic as load_robomimic_trajectories
        if 'obs' in first_demo:
            obs_dict = first_demo['obs']
            obs_components = []
            
            dataset_obs_keys = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
                'robot0_gripper_qvel',
                'robot0_joint_pos_cos',
                'robot0_joint_pos_sin',
                'robot0_joint_vel',
            ]
            
            for key in dataset_obs_keys:
                if key in obs_dict:
                    obs_components.append(obs_dict[key][0])
            
            obs = np.concatenate(obs_components)
            info['obs_dim'] = obs.shape[0]
        else:
            raise ValueError("Could not find 'obs' group in dataset")
        
        action = first_demo['actions'][:]
        info['action_dim'] = action.shape[-1]
        info['num_demos'] = len(f['data'].keys())

        return info


def evaluate_policy_robomimic(
    env: RobomimicEnvWrapper,
    agent,
    num_episodes: int = 20,
    deterministic: bool = True,
    verbose: bool = True,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a policy on robomimic environment.
    
    Args:
        env: Robomimic environment wrapper
        agent: Policy agent with get_action(obs, deterministic) method
        num_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        verbose: Print evaluation progress
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    results = {
        'success_rate': 0.0,
        'avg_steps': 0.0,
        'avg_reward': 0.0,
        'episodes': [],
    }
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        total_reward = 0.0
        steps = 0
        done = False
        images = []

        while not done:
            # Get action from policy
            action = agent.get_action(obs, deterministic=deterministic)
            if isinstance(action, tuple):
                action = action[0]
            if visualize:
                images.append(env.render())

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Record episode results
    
        # success = reward == 1
        success = info['success']
        episode_result = {
            'episode': ep,
            'steps': steps,
            'reward': total_reward,
            'success': success,
            'images': images
        }
        results['episodes'].append(episode_result)
        
        results['success_rate'] += success
        results['avg_steps'] += steps
        results['avg_reward'] += total_reward
        
        if verbose:
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Steps: {steps} | Reward: {total_reward:.2f} | "
                  f"Success: {success}")
    
    # Compute averages
    results['success_rate'] /= num_episodes
    results['avg_steps'] /= num_episodes
    results['avg_reward'] /= num_episodes
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({num_episodes} episodes):")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Avg Steps: {results['avg_steps']:.1f}")
        print(f"  Avg Reward: {results['avg_reward']:.2f}")
        print(f"{'='*60}")
    
    return results


def demo_robomimic_env():
    """
    Demo script to test robomimic environment and dataset loading.
    """
    print("=" * 60)
    print("Robomimic Environment Demo")
    print("=" * 60)
    
    try:
        # Create environment
        print("\n1. Creating Lift environment...")
        env = RobomimicEnvWrapper(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=500
        )
        
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test random rollout
        print("\n2. Testing random policy rollout...")
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 20:  # Limit to 20 steps for demo
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
        
        print(f"   Completed {step} steps")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Success: {info.get('success', False)}")
        
        env.close()
        
        # Test dataset loading
        print("\n3. Dataset Loading Example")
        print("   To load a dataset, use:")
        print("   trajectories = load_robomimic_trajectories(")
        print("       dataset_path='./datasets/low_dim_v141.hdf5',")
        print("       num_trajectories=10")
        print("   )")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed robomimic and robosuite.")
        print("See ROBOMIMIC_SETUP.md for installation instructions.")


if __name__ == "__main__":
    demo_robomimic_env()
