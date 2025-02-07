# python scripts/skrl/train.py --task=Isaac-camera-drone --algorithm PPO  --enable_cameras  --num_envs=100 --ml_framework torch
# Only work with skrl 1.4

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.sensors import (Camera,CameraCfg,RayCaster,RayCasterCfg,TiledCamera,TiledCameraCfg,ContactSensor,ContactSensorCfg)
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

@configclass 
class CrazyflieEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 600.0
    action_scale = 100.0
    debug_vis = True
    action_space = 4
    state_space = 0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=16, replicate_physics=True)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=40,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(3.0, 4.0),
                    platform_width=1.5,
                )
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        debug_vis=False,
    )


    robot = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(1,1,1)))

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Cam", 
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),  
        width=80,
        height=80,
    )

    num_channels = 3
    observation_space = {
    "camera": [camera.width, camera.height, num_channels],
    }
    
    thrust_to_weight = 1.9
    moment_scale = 0.01


class CrazyflieEnv(DirectRLEnv):

    cfg: CrazyflieEnvCfg
    def __init__(self, cfg: CrazyflieEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
    
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Initialize marker positions once
        self.red_marker_pos = torch.zeros(1, 3, device=self.device).uniform_(-self.scene.cfg.env_spacing/2, self.scene.cfg.env_spacing/2)
        self.blue_marker_pos = torch.zeros(1, 3, device=self.device).uniform_(-self.scene.cfg.env_spacing/2, self.scene.cfg.env_spacing/2)
        self.green_marker_pos = torch.zeros(1, 3, device=self.device).uniform_(-self.scene.cfg.env_spacing/2, self.scene.cfg.env_spacing/2)
        self.red_marker_pos[:, 2] = torch.zeros(1, device=self.device).uniform_(2, self.scene.cfg.env_spacing-2)
        self.blue_marker_pos[:, 2] = torch.zeros(1, device=self.device).uniform_(2, self.scene.cfg.env_spacing-2)
        self.green_marker_pos[:, 2] = torch.zeros(1, device=self.device).uniform_(2, self.scene.cfg.env_spacing-2)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._camera = self.cfg.camera.class_type(self.cfg.camera)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)


    def _get_observations(self) -> dict:
        cam_images = self._camera.data.output["rgb"]

        observation = {
            "policy": {
                "camera": cam_images,
            }
        }
        return observation

    def _get_rewards(self) -> torch.Tensor:

        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Altitude constraints
        low_altitude = self._robot.data.root_pos_w[:, 2] < 0.1
        high_altitude = self._robot.data.root_pos_w[:, 2] > self.scene.cfg.env_spacing
        altitude = torch.logical_or(low_altitude, high_altitude)

        # X-Y Plane constraints
        x_constraint = torch.logical_or(self._robot.data.root_pos_w[:, 0] < -self.scene.cfg.env_spacing/2, self._robot.data.root_pos_w[:, 0] > self.scene.cfg.env_spacing/2)
        y_constraint = torch.logical_or(self._robot.data.root_pos_w[:, 1] < -self.scene.cfg.env_spacing/2, self._robot.data.root_pos_w[:, 1] > self.scene.cfg.env_spacing/2)
        xy_constraint = torch.logical_or(x_constraint, y_constraint)

        # Domain constraints
        domain_constraint = torch.logical_or(xy_constraint, altitude)

        died = domain_constraint
        time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: #type:ignore
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # Randomize drone positions
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 0] += torch.zeros_like(default_root_state[:, 0]).uniform_(-self.scene.cfg.env_spacing/2, self.scene.cfg.env_spacing/2)
        default_root_state[:, 1] += torch.zeros_like(default_root_state[:, 1]).uniform_(-self.scene.cfg.env_spacing/2, self.scene.cfg.env_spacing/2)
        default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(2, self.scene.cfg.env_spacing-2)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "red_marker_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/red_marker"
                self.red_marker_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.red_marker_visualizer.set_visibility(True)

            if not hasattr(self, "blue_marker_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/blue_marker"
                self.blue_marker_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.blue_marker_visualizer.set_visibility(True)

            if not hasattr(self, "green_marker_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/green_marker"
                self.green_marker_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.green_marker_visualizer.set_visibility(True)
        else:
            if hasattr(self, "red_marker_visualizer"):
                self.red_marker_visualizer.set_visibility(False)
            if hasattr(self, "blue_marker_visualizer"):
                self.blue_marker_visualizer.set_visibility(False)
            if hasattr(self, "green_marker_visualizer"):
                self.green_marker_visualizer.set_visibility(False)
                

    def _debug_vis_callback(self, event):
        # Update the markers
        self.red_marker_visualizer.visualize(self.red_marker_pos.repeat(self.num_envs, 1))
        self.blue_marker_visualizer.visualize(self.blue_marker_pos.repeat(self.num_envs, 1))
        self.green_marker_visualizer.visualize(self.green_marker_pos.repeat(self.num_envs, 1))

