# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
torch.set_printoptions(sci_mode=False, precision=10)
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from inverse_kinematics import inverse_kinematics

from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

from isaaclab_assets import UR10_CFG  # isort:skip
from kinematics import forward_kinematics as robot_forward_kinematics

@configclass
class ur10SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    target_quat = [0 , -0.7071, 0.7071, 0.0]
    target_pos = [0.6, 0.6, 0.3]
    robot = scene["robot"]
    print(robot.joint_names)
    print(robot.body_names)
    tool_frame_id = robot.body_names.index("ee_link")
    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    base_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/base_frame"))

    # Define goals for the arm
    base_frame = [0.0, 0.0, 0, 1, 0, 0, 0]
    base_frame = torch.tensor(base_frame, device=sim.device, dtype=torch.float32).unsqueeze(0)
    sim_dt = sim.get_physics_dt()
    count = 0
    angles_rad = torch.tensor(inverse_kinematics(target_pos, target_quat))
    
    # Simulation loop
    while simulation_app.is_running():
        robot.set_joint_position_target(angles_rad)
        print(count)
        if count % 150 == 0:
            print("#" * 15)
            print("[INFO]: End effector position Feedback from simulation", robot.data.body_link_pos_w[0, tool_frame_id, :])
            print("[INFO]: End effector orientation Feedback from simulation", robot.data.body_link_quat_w[0, tool_frame_id, :])
            ja = ((robot.data.joint_pos)[0]).cpu().numpy()
            print("[INFO]: Joint angles feedback from simulation", ja * 180.0 / torch.pi)
            # position = robot_forward_kinematics(ja)
            # print("[INFO]: Adjusted End effector position based on joint angles feedback from simulation")
            # print(position)
        robot.write_data_to_sim()
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        base_marker.visualize(base_frame[:,0:3], base_frame[:,3:7])
        


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = ur10SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
