# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator
and have the UR10 end effector "draw" its path live by updating a USD BasisCurves prim each frame.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Tutorial on using Absolute IK with live drawing")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from dev.inverse_kinematics import inverse_kinematics
from isaaclab_assets import UR10_CFG  # isort:skip

# USD imports for live drawing
from omni.usd import get_context
from pxr import UsdGeom, Gf, Sdf

torch.set_printoptions(sci_mode=False, precision=10)

@configclass
class ur10SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, basis: UsdGeom.BasisCurves, max_steps: int):
    """Runs the simulation loop and records the EE path live into the USD BasisCurves"""

    global target_pos, target_quat, goal_id, traj_x, traj_y, traj_z
    robot = scene["robot"]
    tool_frame_id = robot.body_names.index("ee_link")
    sim_dt = sim.get_physics_dt()

    points_recorded = []
    step = 0
    angles_rad = torch.tensor(inverse_kinematics(target_pos, target_quat), device=sim.device)
    state_error_count = 0
    prv_dist = 0
    tolerance = 0.02
    j1 = []
    j2 = []
    j3 = []
    j4 = []
    j5 = []
    j6 = []

    while simulation_app.is_running():
        robot.set_joint_position_target(angles_rad)
        robot.write_data_to_sim()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        simulation_app.update()
        ee_pos_np = robot.data.body_link_pos_w[0, tool_frame_id, :].cpu().numpy()
        dist = np.linalg.norm(ee_pos_np - np.array(target_pos))
        
        s_dist = np.linalg.norm(dist - prv_dist)
        print(step,goal_id,s_dist,state_error_count,prv_dist,dist)

        if s_dist<0.001:
            state_error_count+=1
        prv_dist = dist
        
        if state_error_count == 10:
            tolerance += 0.005
            state_error_count = 0

        if dist < tolerance:
            goal_id +=1
            target_pos = [traj_x[goal_id], traj_y[goal_id], traj_z[goal_id]]
            angles_rad = torch.tensor(inverse_kinematics(target_pos, target_quat), device=sim.device)

        x, y, z = float(ee_pos_np[0]), float(ee_pos_np[1]), float(ee_pos_np[2])
        if goal_id>0:
            points_recorded.append(Gf.Vec3f(x, y, z))
            ja = ((robot.data.joint_pos)[0]).cpu().numpy()
            ja = ja*180.0/torch.pi

            j1.append(ja[0])
            j2.append(ja[1])
            j3.append(ja[2])
            j4.append(ja[3])
            j5.append(ja[4])
            j6.append(ja[5])

            if len(points_recorded) > max_steps:
                points_recorded.pop(0)
            if len(points_recorded) < max_steps:
                padded = points_recorded + [points_recorded[-1]] * (max_steps - len(points_recorded))
            else:
                padded = points_recorded[:max_steps]
            basis.GetPointsAttr().Set(padded)
            if goal_id == np.shape(traj_z)[0]-1:
                np.save("jointangles/j1.npy", np.array(j1))
                np.save("jointangles/j2.npy", np.array(j2))
                np.save("jointangles/j3.npy", np.array(j3))
                np.save("jointangles/j4.npy", np.array(j4))
                np.save("jointangles/j5.npy", np.array(j5))
                np.save("jointangles/j6.npy", np.array(j6))
                break
        step += 1


def main():
    # Setup sim & scene
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = ur10SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Define IK target
    global target_pos, target_quat, goal_id, traj_x, traj_y, traj_z

    a = 0
    b = 0.01 
    num_turns = 5
    points_per_turn = 25

    theta_spiral = np.linspace(0, 2*np.pi*num_turns, num_turns*points_per_turn)

    r = a + b*theta_spiral
    traj_x = r * np.cos(theta_spiral)
    traj_y = r * np.sin(theta_spiral) + 0.7
    traj_z = (0.15 / (2 * np.pi)) * theta_spiral

    goal_id = 0
    target_pos = [traj_x[goal_id], traj_y[goal_id], traj_z[goal_id]]
    target_quat = [0, -0.7071, 0.7071, 0]

    # USD BasisCurves setup for live drawing
    stage = get_context().get_stage()
    curve_path = Sdf.Path("/World/UR10DrawCurve")
    basis = UsdGeom.BasisCurves.Define(stage, curve_path)
    max_steps = 1000
    basis.CreateCurveVertexCountsAttr().Set([max_steps])  
    basis.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    basis.CreateWidthsAttr().Set([0.01] * max_steps) 
    basis.CreateDisplayColorAttr().Set([(0.0, 1.0, 0.0)])

    run_simulator(sim, scene, basis, max_steps)
    simulation_app.update()
    print("Exiting simulation")
    simulation_app.close()
    print("Exited simulation")
    simulation_app.update()

if __name__ == "__main__":
    main()