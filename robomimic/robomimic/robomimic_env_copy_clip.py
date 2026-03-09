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


    def compute_dense_reward_from_info_not_so_good(self, info: dict) -> float:
        """
        Dense reward encouraging:
        1. Center gripper over cube (XY alignment)
        2. Hover slightly above cube (Z alignment)
        3. Align gripper pointing downward
        4. Close gripper ONLY when centered
        5. Lift cube AFTER proper grasp
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
    
    def compute_dense_reward_from_info_v1_grasp(self, info: dict) -> float:
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

        HOVER_HEIGHT = 0.1                        # target height above cube center
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
        """
        if "robot0_eef_pos" not in info:
            return 0.0

        eef_pos  = np.array(info["robot0_eef_pos"])
        cube_pos = np.array(info.get("cube_pos", info.get("object", None)))
        if cube_pos is None:
            return 0.0

        # ── Phase 1: XY + Z Approach ─────────────────────────────────────────
        xy_dist = float(np.linalg.norm(eef_pos[:2] - cube_pos[:2]))

        HOVER_HEIGHT = 0.1                        # target height above cube center
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
            r_proximity_grasp = float(np.exp(-3.0 * xy_dist)) * gripper_norm


            # Only reward closing when positioned AND oriented correctly
            readiness = approach_score * r_orient    # 0..1
            r_grasp = r_proximity_grasp + readiness * gripper_norm
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


        lift_ready = gripper_norm * r_orient * r_xy  # i.e., ready when gripping, oriented, close
        r_lift = lift_ready * float(np.tanh(10.0 * max(0.0, height_above_table)))
        if height_above_table > 0.05:
            r_lift += 1.0

        # ── Weighted Sum ──────────────────────────────────────────────────────
        reward = (
            2.0  * r_xy        +   # always guide toward cube XY
            1.0  * r_z         +   # guide Z hover
            1.0  * r_orient    +   # orientation always encouraged
            4.0  * r_grasp     +   # grasp gated by approach quality
            10.0 * r_lift          # lift is the main task signal
        )

        return float(reward)
        
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

        reward =  self.compute_dense_reward_from_info(obs_dict)   #co
            
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
    
        success = reward == 1
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
