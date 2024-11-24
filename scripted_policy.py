import IPython
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]["t"] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        box_info = np.array(ts_first.observation["env_state"])
        box_xyz = box_info[:3]
        # box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 100,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # approach meet position
            {
                "t": 260,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 1,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz + np.array([0.02, 0, -0.02]),
                "quat": meet_left_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # move left
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.1, 0, -0.02]),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            },  # stay
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 90,
                "xyz": box_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 130,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 170,
                "xyz": box_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 200,
                "xyz": meet_xyz + np.array([0.05, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 220,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # move to meet position
            {
                "t": 310,
                "xyz": meet_xyz,
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # open gripper
            {
                "t": 360,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # move to right
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.1, 0, 0]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # stay
        ]


class InsertionPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation["mocap_pose_right"]
        init_mocap_pose_left = ts_first.observation["mocap_pose_left"]

        peg_info = np.array(ts_first.observation["env_state"])[:7]
        peg_xyz = peg_info[:3]
        # peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation["env_state"])[7:]
        socket_xyz = socket_info[:3]
        # socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=60
        )

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_left[:3],
                "quat": init_mocap_pose_left[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": socket_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": socket_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([-0.1, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([-0.05, 0, 0]),
                "quat": gripper_pick_quat_left.elements,
                "gripper": 0,
            },  # insertion
        ]

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose_right[:3],
                "quat": init_mocap_pose_right[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 120,
                "xyz": peg_xyz + np.array([0, 0, 0.08]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 170,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 220,
                "xyz": peg_xyz + np.array([0, 0, -0.03]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # close gripper
            {
                "t": 285,
                "xyz": meet_xyz + np.array([0.1, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # approach meet position
            {
                "t": 340,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
            {
                "t": 400,
                "xyz": meet_xyz + np.array([0.05, 0, lift_right]),
                "quat": gripper_pick_quat_right.elements,
                "gripper": 0,
            },  # insertion
        ]


class MoveCubeToPlatePolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose = ts_first.observation["mocap_pose"]

        # 获取物体位置
        env_state = np.array(ts_first.observation["env_state"])
        cube_pose = env_state[:7]  # 立方体的完整位姿
        plate_pose = env_state[7:]  # 板子的完整位姿

        cube_xyz = cube_pose[:3]  # 立方体位置
        plate_xyz = plate_pose[:3]  # 板子位置

        # 设置抓取姿态（参考PickAndTransferPolicy）
        gripper_quat = Quaternion(init_mocap_pose[3:])
        gripper_quat = gripper_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose[:3],
                "quat": init_mocap_pose[3:],
                "gripper": 0,  # 开始时闭合夹爪
            },
            {
                "t": 90,
                "xyz": cube_xyz + np.array([0, 0, 0.08]),  # 使用相同的预备高度
                "quat": gripper_quat.elements,
                "gripper": 1,  # 打开夹爪
            },
            {
                "t": 130,
                "xyz": cube_xyz + np.array([0, 0, -0.015]),  # 使用相同的抓取高度
                "quat": gripper_quat.elements,
                "gripper": 1,
            },
            {
                "t": 170,
                "xyz": cube_xyz + np.array([0, 0, -0.015]),
                "quat": gripper_quat.elements,
                "gripper": 0,  # 闭合夹爪
            },
            {
                "t": 220,
                "xyz": cube_xyz + np.array([0, 0, 0.08]),  # 提起到预备高度
                "quat": gripper_quat.elements,
                "gripper": 0,
            },
            {
                "t": 280,
                "xyz": plate_xyz + np.array([0, 0, 0.08]),  # 移动到板子上方
                "quat": gripper_quat.elements,
                "gripper": 0,
            },
            {
                "t": 320,
                "xyz": plate_xyz + np.array([0, 0, 0.03]),  # 下降到放置高度
                "quat": gripper_quat.elements,
                "gripper": 0,
            },
            {
                "t": 360,
                "xyz": plate_xyz + np.array([0, 0, 0.03]),
                "quat": gripper_quat.elements,
                "gripper": 1,  # 释放
            },
            {
                "t": 400,
                "xyz": plate_xyz + np.array([0, 0, 0.15]),  # 抬起
                "quat": gripper_quat.elements,
                "gripper": 1,
            },
        ]

        # 为了兼容基类的逻辑，我们创建一个空的左臂轨迹
        self.left_trajectory = [
            {
                "t": t,
                "xyz": np.zeros(3),
                "quat": np.array([1, 0, 0, 0]),
                "gripper": 0,
            }
            for t in [0, 400]
        ]

    def __call__(self, ts):
        # 重写父类的 __call__ 方法，只返回右臂（主要机械臂）的动作
        if self.step_count == 0:
            self.generate_trajectory(ts)

        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        if self.inject_noise:
            scale = 0.01
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return action


def test_policy(task_name):
    onscreen_render = True
    inject_noise = False

    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]
    if task_name == "sim_move_cube_to_plate":
        env = make_ee_sim_env("sim_move_cube_to_plate")
        policy_class = MoveCubeToPlatePolicy
    elif "sim_transfer_cube" in task_name:
        env = make_ee_sim_env("sim_transfer_cube")
        policy_class = PickAndTransferPolicy
    elif "sim_insertion" in task_name:
        env = make_ee_sim_env("sim_insertion")
        policy_class = InsertionPolicy
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["images"]["angle"])
            plt.ion()

        policy = policy_class(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation["images"]["angle"])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == "__main__":
    test_task_name = "sim_move_cube_to_plate"  # 修改默认测试任务
    test_policy(test_task_name)
