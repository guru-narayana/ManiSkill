from typing import Any, Dict, Union

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

TRAJECTORY_TIME = 0.5 # max_episode_steps = 100
MAX_EPISODE_STEPS = max(TRAJECTORY_TIME//0.02 + 10, 50) # 10 steps for grasping and remaining steps for tossing
SAMPLING_INDICES = TRAJECTORY_TIME/0.02

@register_env("TossCube-v1", max_episode_steps= MAX_EPISODE_STEPS)
class TossCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]

    agent: Union[Panda, Xmate3Robotiq, Fetch]

    goal_radius = 0.06
    goal_height = 0.03
    cube_side = 0.022
    toss_start_location: torch.Tensor # location where the cube is tossed from

    # reward weights
    REACH_STAGE_WEIGHT: float = 1.0
    TOSS_STAGE_WEIGHT: float = 1.0
    IN_AIR_STAGE_WEIGHT: float = 10.0
    # stage completion bonuses
    grasped_bonus: float =  0.0
    toss_stage_bonus: float = 00.0
    # Status Variables
    grasp_complete: torch.Tensor # status of stage 1 : grasping the cube 
    toss_reach_complete: torch.Tensor # status of stage 2 : reaching the toss start location

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        self.grasp_complete = torch.zeros(kwargs["num_envs"], dtype=torch.float32)
        self.toss_reach_complete = torch.zeros(kwargs["num_envs"], dtype=torch.float32)

        self.trajectory_progress = torch.zeros(kwargs["num_envs"], dtype=torch.float32)
        
        self.toss_start_location = torch.zeros(kwargs["num_envs"], 3, dtype=torch.float32)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.7, 1.1, 0.6], [0.7, -0.2, 0.0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_side,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )
        
        self.goal_region = actors.build_hollow_cylinder(
            scene=self.scene,
            outer_radius=self.goal_radius + 0.01,
            inner_radius=self.goal_radius,
            height=self.goal_height,
            inner_color=[0, 0, 0, 0.5],
            outer_color=[0.6, 0.1, 0.1, 1],
            name="basket_goal",
            body_type="kinematic",
            add_collision=False
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        
        self._move_all_to_device(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.robot_pose_mine = Pose.create_from_pq(p = [0.0, 0, 0])
            self.agent.robot.set_pose(self.robot_pose_mine)

            xyz = torch.zeros((b, 3))
            xyz[...,  0] = 0.4
            xyz[..., 1] = 0.0
            xyz[..., 2] = self.cube_side
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(obj_pose)

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[...,  0] = 0.6 + torch.rand(b) * 0.1
            xyz_goal[..., 1]  = 0.0
            xyz_goal[..., 2]  = -0.45
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )


        self.grasp_complete[env_idx] = 0  # reset the grasp complete flag
        self.trajectory_progress[env_idx] = 0  # reset the trajectory progress



        self.true_goal = self.goal_region.pose.p + torch.tensor([0, 0, self.goal_height], device=self.device)

        self.toss_start_location[env_idx] = self.cube.pose.p.clone()[env_idx] + torch.tensor([0.4, 0, 0.3], device=self.device)

    def _move_all_to_device(self,device):
        if self.grasp_complete.device != device:
            self.grasp_complete = self.grasp_complete.to(device)
            self.toss_start_location = self.toss_start_location.to(device)
            self.trajectory_progress = self.trajectory_progress.to(device)
            self.toss_reach_complete = self.toss_reach_complete.to(device)

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        is_height_correct = (
            self.cube.pose.p[..., 2] - self.true_goal[..., 2]
        ) < 0.01 + self.cube_side

        success = is_obj_placed & is_height_correct
        fail    = ~is_obj_placed & is_height_correct

        mse_to_minpoint, min_point, minpoint_velocity = self._traj_requirments(self.cube.pose.p)
        
        return {
            "success": success,
            "fail": fail,
            "mse_to_minpoint": mse_to_minpoint,
            "min_point": min_point,
            "minpoint_velocity": minpoint_velocity,
        }
    
    def _get_obs_extra(self, info: Dict):

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            #TODO: change the observations here
            obs.update(
                cube_pose                 = self.cube.pose.p,
                tcp_to_cube_pos           = self.cube.pose.p - self.agent.tcp.pose.p,
                
                # mse_to_minpoint           = info["mse_to_minpoint"],
                # min_point                 = info["min_point"],
                # cube_vel                  = self.cube.linear_velocity,
                # cube_required_vel         = info["minpoint_velocity"],
                # cube_vel_diff             = info["minpoint_velocity"] - self.cube.linear_velocity,
                
                toss_start_pose = self.toss_start_location,
                toss_rel_start_pose = self.cube.pose.p -  self.toss_start_location,
                
                goal_pos         =  self.true_goal,
                cube_to_goal_pos = self.cube.pose.p - self.true_goal,


                #grasp_complete = self.grasp_complete,
                #toss_reach_complete = self.toss_reach_complete,
                #toss_complete = self.toss_complete,
            )
        return obs
    
    def _calculate_initial_velocity_trajectory(self, dx, dy, dz,g = 9.8):
        v0x = dx / TRAJECTORY_TIME
        v0y = dy / TRAJECTORY_TIME
        v0z = (dz + 0.5 * g * (TRAJECTORY_TIME**2)) / TRAJECTORY_TIME
        return torch.stack([v0x, v0y, v0z], dim=1)

    def _traj_requirments(self, point, g=9.8, num_samples=SAMPLING_INDICES):

        t_values = ((1/num_samples)*self.trajectory_progress*TRAJECTORY_TIME)[:,None]
        dx = self.true_goal[..., 0] - self.toss_start_location[:,0]
        dy = self.true_goal[..., 1] - self.toss_start_location[:,1]
        dz = self.true_goal[..., 2] - self.toss_start_location[:,2]
        vel = self._calculate_initial_velocity_trajectory(dx, dy, dz)

        trajectory_points = self.toss_start_location + vel * t_values
        trajectory_points[..., 2] -= 0.5 * g * (t_values[..., 0] ** 2)
        
        self.trajectory_progress += self.grasp_complete
        self.trajectory_progress[self.trajectory_progress > num_samples] = num_samples

        min_distances = torch.linalg.norm(trajectory_points - point, dim=1)
        
        dx = self.true_goal[..., 0] - trajectory_points[:,0]
        dy = self.true_goal[..., 1] - trajectory_points[:,1]
        dz = self.true_goal[..., 2] - trajectory_points[:,2]
        req_vel = self._calculate_initial_velocity_trajectory(dx, dy, dz)

        return min_distances, trajectory_points, req_vel

    def _distance_to_reward(self, distance , mode = "tanh", scale = 1.0):
        if mode == "inverse":
            return 1/(1+scale*distance)
        elif mode == "tanh":
            return (1 - torch.tanh(scale*distance))
        else:
           raise NotImplementedError(f"{mode} mode for distance to reward conversion is not implemented.")

    def _reach_stage_reward(self,info: Dict):
        tcp_reach_pose = Pose.create_from_pq(p=self.cube.pose.p)
        tcp_to_reach_pose = tcp_reach_pose.p - self.agent.tcp.pose.p
        grasped = self.agent.is_grasping(self.cube).to(torch.float32)
        tcp_to_reach_pose_dist = torch.linalg.norm(tcp_to_reach_pose, axis=1) # distance between tcp and cube        
        reaching_reward = self.REACH_STAGE_WEIGHT*self._distance_to_reward(tcp_to_reach_pose_dist,mode="tanh") # reward for reaching the cube
        reaching_reward*=(1-self.grasp_complete) # only give reward if the cube is not grasped
        reaching_reward += self.grasped_bonus * grasped *(1-self.grasp_complete) # give bonus if the cube is grasped
        self.grasp_complete[self.agent.is_grasping(self.cube)] = 1 # set the grasp complete flag to 1 if the cube is grasped
        return reaching_reward
    

    def _ready_stage_reward(self,info: Dict):

        cube_to_toss_pose = self.toss_start_location - self.cube.pose.p
        cube_to_toss_pose_dist = torch.linalg.norm(cube_to_toss_pose, axis=1) # distance between tcp and toss location
        pos_reaching_reward = self.TOSS_STAGE_WEIGHT*self._distance_to_reward(cube_to_toss_pose_dist) # reward for reaching the toss location
        grasped = self.agent.is_grasping(self.cube).to(torch.float32)

        pos_reaching_reward *=  grasped * \
                                self.grasp_complete * \
                                (1 - self.toss_reach_complete) # only give reward if the cube is grasped and not reached the toss location
        
        toss_reached = torch.logical_and(self.grasp_complete.to(torch.bool),(cube_to_toss_pose_dist < 0.05)) # check if the toss location is reached
        pos_reaching_reward += self.toss_stage_bonus * \
                                grasped * \
                                toss_reached.to(torch.float32) * \
                                (1 - self.toss_reach_complete)            # give bonus if the cube is grasped and reached the toss location
        pos_reaching_reward += self.grasp_complete                        # reward for being in a better position to toss the cube
        
        self.toss_reach_complete[toss_reached] = grasped[toss_reached] # set the toss reach complete flag to 1 if the toss location is reached
        return pos_reaching_reward


    def _in_air_reward(self,info: Dict):

        not_grasped = 1 - self.agent.is_grasping(self.cube).to(torch.float32)
        cube_goal_dist = torch.linalg.norm(self.cube.pose.p[...,0:2]- self.true_goal[...,0:2], axis=1)

        vel_diff = info["minpoint_velocity"] - self.cube.linear_velocity

        vel_reward_xy = 1 - torch.tanh(torch.linalg.norm(vel_diff[...,:2], axis=1)) # reward for matching the velocity
        vel_reward_z  = 1 - torch.tanh(torch.abs(vel_diff[...,2]))                  # reward for matching the velocity

        vel_reward = vel_reward_xy #+ 0.4*vel_reward_z

        cube_to_traj_dist = info["mse_to_minpoint"]

        air_traj_reward_reach  = self._distance_to_reward(cube_to_traj_dist,scale=1) # reward for reaching the toss location
        air_traj_reward = air_traj_reward_reach.clone()
        #air_traj_reward += vel_reward
        air_traj_reward *=  self.IN_AIR_STAGE_WEIGHT \
                                *self.toss_reach_complete \
                                *(1+3*not_grasped) \
        
        
        air_traj_reward += (2+not_grasped)*self.toss_reach_complete


        air_traj_reward[info["success"]] = 50
        air_traj_reward[info["fail"]]    = 20*air_traj_reward_reach[info["fail"]]*self.toss_reach_complete[info["fail"]]

        
        
        return air_traj_reward,air_traj_reward_reach

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        reach_reward = self._reach_stage_reward(info)
        toss_reward,val = self._in_air_reward(info)
        reached_reward =self._ready_stage_reward(info)
        return reach_reward + toss_reward + reached_reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 50.0
