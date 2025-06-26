# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="script to visualize dh frames.")
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
from dev.kinematics import forward_kinematics,joint_pos
##
# Pre-defined configs
##
from isaaclab_assets import UR10_CFG  # isort:skip


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

    robot = scene["robot"]
    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    base_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/base_frame"))
    j1 = VisualizationMarkers(  frame_marker_cfg.replace(prim_path=f"/Visuals/j1_frame"))
    j2 = VisualizationMarkers(  frame_marker_cfg.replace(prim_path=f"/Visuals/j2_frame"))
    j3 = VisualizationMarkers(  frame_marker_cfg.replace(prim_path=f"/Visuals/j3_frame"))
    j4 = VisualizationMarkers(  frame_marker_cfg.replace(prim_path=f"/Visuals/j4_frame"))
    j5 = VisualizationMarkers(  frame_marker_cfg.replace(prim_path=f"/Visuals/j5_frame"))
    end_effector_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_frame"))

    angles = [9.57625294, -130.99350741,  156.5652118,  -115.57170439,  -90.0, 9.57625294]

    # Define goals for the arm
    base_frame = [0.0, 0.0, 0.0, 1, 0, 0, 0]
    base_frame = torch.tensor(base_frame, device=sim.device, dtype=torch.float32).unsqueeze(0)
    j1_frame = torch.tensor(joint_pos(angles,1), device=sim.device, dtype=torch.float32).unsqueeze(0)
    j2_frame = torch.tensor(joint_pos(angles,2), device=sim.device, dtype=torch.float32).unsqueeze(0)
    j3_frame = torch.tensor(joint_pos(angles,3), device=sim.device, dtype=torch.float32).unsqueeze(0)
    j4_frame = torch.tensor(joint_pos(angles,4), device=sim.device, dtype=torch.float32).unsqueeze(0)
    j5_frame = torch.tensor(joint_pos(angles,5), device=sim.device, dtype=torch.float32).unsqueeze(0)

    tool_frame = torch.tensor(forward_kinematics(angles), device=sim.device, dtype=torch.float32).unsqueeze(0)

    print(base_frame)
    print(j2_frame)
    print(j3_frame)
    print(j4_frame)
    print(j5_frame)
    print(tool_frame)

    sim_dt = sim.get_physics_dt()
    count = 0
    angles_rad = torch.tensor(angles, device=sim.device)*torch.pi/180.0
    # Simulation loop
    while simulation_app.is_running():
        robot.set_joint_position_target(angles_rad)
        robot.write_data_to_sim()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count+=1
        base_marker.visualize(base_frame[:,0:3], base_frame[:,3:7])
        j1.visualize(j1_frame[:,0:3], j1_frame[:,3:7])
        j2.visualize(j2_frame[:,0:3], j2_frame[:,3:7])
        j3.visualize(j3_frame[:,0:3], j3_frame[:,3:7])
        j4.visualize(j4_frame[:,0:3], j4_frame[:,3:7])
        j5.visualize(j5_frame[:,0:3], j5_frame[:,3:7])
        end_effector_marker.visualize(tool_frame[:,0:3], tool_frame[:,3:7])
        


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
