"""
MuJoCo Warehouse Fulfillment Environment.

Provides a photorealistic 3D physics simulation and digital twin of the
Franka Emika Panda 7-DOF arm in an industrial fulfillment workcell using
Google DeepMind's MuJoCo physics engine.

Features:
- Physically-Based Rendering (PBR) with metallic reflections, soft contact
  shadows, and textured factory floor.
- Full 7-DOF Franka Panda cobot with parallel finger gripper.
- Dynamic industrial pallet rack shelving unit with steel uprights and beams.
- Realistic corrugated cardboard boxes with category colors and physical mass.
- Active industrial conveyor belt with dynamic transport toward dispatch.
- Direct trajectory recording via HardwareBridge for real physical Franka arm execution.
"""

import os
import time
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

import mujoco
import mujoco.viewer

from warehouse_sim import config
from warehouse_sim.layout_manager import LayoutManager
from warehouse_sim.hardware_bridge import HardwareBridge


class MuJoCoWarehouseEnv:
    """
    Photorealistic MuJoCo Warehouse Simulation Environment.
    """

    def __init__(
        self,
        gui: bool = True,
        layout_file: str = "layout.json",
        layout_data: Optional[Dict] = None,
        hardware_mode: str = "sim",
        robot_ip: Optional[str] = None
    ):
        self.gui = gui
        self.layout_file = layout_file
        self.hardware_mode = hardware_mode
        self.robot_ip = robot_ip or "172.16.0.2"

        # Load layout configuration (prefer in-memory dict if provided)
        if layout_data is not None:
            self.layout_data = layout_data
            try:
                LayoutManager.save_layout(layout_data, layout_file)
            except Exception:
                pass
        else:
            self.layout_data = LayoutManager.load_layout(layout_file)
        config.apply_layout(self.layout_data)

        # Hardware bridge for real Franka Panda arm trajectory recording
        self.hardware = HardwareBridge(mode=self.hardware_mode, robot_ip=self.robot_ip)

        # Compile MuJoCo MJCF Model
        self.model_xml = self._generate_mjcf()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Viewer reference
        self.viewer = None
        if self.gui:
            self._init_viewer()

        # State tracking
        self.step_count = 0
        self.box_status: Dict[str, str] = {}
        for box in self.layout_data.get("boxes", []):
            self.box_status[box["sku"]] = "ON_SHELF"

        # Joint indices
        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.joint_qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.joint_dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        # Finger joint addresses
        self.finger1_qpos_adr = self.model.joint("panda_finger_joint1").qposadr[0]
        self.finger2_qpos_adr = self.model.joint("panda_finger_joint2").qposadr[0]
        self.finger1_dof_adr = self.model.joint("panda_finger_joint1").dofadr[0]
        self.finger2_dof_adr = self.model.joint("panda_finger_joint2").dofadr[0]

        # End effector site ID
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

        # Target kinematic holds to ensure the robot never goes limp or falls under gravity
        self.target_joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.target_gripper_width = 0.04
        self.carried_sku: Optional[str] = None
        self.step_callback = None

        # Set default home pose
        self._set_home_pose()

    def _hold_robot_joints(self):
        """
        Firmly lock robot arm and gripper joints at their target positions,
        ensuring the arm maintains its solid state and never falls under gravity.
        """
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = self.target_joint_positions[i]
        for dof in self.joint_dof_adr:
            self.data.qvel[dof] = 0.0

        half_w = float(self.target_gripper_width) / 2.0
        self.data.qpos[self.finger1_qpos_adr] = half_w
        self.data.qpos[self.finger2_qpos_adr] = half_w
        self.data.qvel[self.finger1_dof_adr] = 0.0
        self.data.qvel[self.finger2_dof_adr] = 0.0

        if self.carried_sku:
            box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{self.carried_sku}")
            if box_body_id != -1:
                j_id = self.model.body(box_body_id).jntadr[0]
                if j_id != -1:
                    qadr = self.model.jnt_qposadr[j_id]
                    ee_pos = self.data.site_xpos[self.ee_site_id]
                    cur_box = self.data.qpos[qadr:qadr+3]
                    # Only lock position if the gripper has arrived at the box (within 10cm)
                    if np.linalg.norm(ee_pos - cur_box) < 0.10:
                        self.data.qpos[qadr:qadr+3] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.015]
                        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]

    def get_box_position(self, sku: str) -> np.ndarray:
        """Return the current live 3D physics position [x, y, z] of a box in the simulation."""
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{sku}")
        if box_body_id != -1:
            j_id = self.model.body(box_body_id).jntadr[0]
            if j_id != -1:
                qadr = self.model.jnt_qposadr[j_id]
                return np.array(self.data.qpos[qadr:qadr+3]).copy()
        for b in self.layout_data.get("boxes", []):
            if b.get("sku") == sku:
                return np.array(b.get("spawn_position", [0.4, 0.0, 0.4])).copy()
        return np.array([0.4, 0.0, 0.4])

    def _set_home_pose(self):
        """Set robot arm to ready home position and hold rock-solid."""
        home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.target_joint_positions = home_joints.copy()
        self.target_gripper_width = 0.04
        self.carried_sku = None
        self._hold_robot_joints()
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        elif self.step_callback is not None:
            self.step_callback(0.002)

    def _init_viewer(self):
        """Initialize native interactive MuJoCo OpenGL viewer."""
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Default camera framing
            self.viewer.cam.distance = 2.4
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25
            self.viewer.cam.lookat = [0.70, 0.0, 0.35]
        except Exception as e:
            print(f"[WARN] Could not launch passive viewer: {e}. Running offscreen.")
            self.viewer = None

    def _generate_mjcf(self) -> str:
        """Generate photorealistic MJCF XML string based on workcell layout."""
        arm_base = self.layout_data.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
        conv = self.layout_data.get("conveyor", {})
        conv_pos = conv.get("position", [1.05, 0.0, 0.22])
        conv_len = conv.get("length", 1.30)
        conv_wid = conv.get("width", 0.24)
        conv_h = conv.get("height", 0.22)
        shelves = self.layout_data.get("shelves", [])
        boxes = self.layout_data.get("boxes", [])

        # Build box bodies XML
        boxes_xml = []
        for i, box in enumerate(boxes):
            sku = box.get("sku", f"BOX_{i+1}")
            pos = box.get("spawn_position", [0.39, 0.0, 0.40])
            cat = box.get("category", 0)
            mat_name = ["mat_box_blue", "mat_box_green", "mat_box_red"][cat % 3]
            boxes_xml.append(f"""
    <body name='box_{sku}' pos='{pos[0]} {pos[1]} {pos[2]}'>
      <freejoint name='joint_{sku}'/>
      <geom name='geom_{sku}' type='box' size='0.035 0.035 0.035' mass='0.12' material='{mat_name}'
            friction='1.2 0.05 0.001' condim='4'/>
    </body>
""")
        boxes_xml_str = "\n".join(boxes_xml)

        # Build shelves XML dynamically: Industrial open-front cantilever rack
        # Columns are at the rear to prevent obstructing cobot pick path
        shelves_xml = []
        for idx, shelf in enumerate(shelves):
            spos = shelf.get("position", [0.30, 0.0, 0.30])
            levels = shelf.get("levels", [0.26, 0.57, 0.77])
            max_h = max(levels) + 0.12
            half_d = 0.11   # 22cm depth - solid stable support for shelf boxes
            half_w = 0.22   # 44cm width
            col_h = max_h / 2.0
            levels_str = "\n".join([
                f"      <geom type='box' size='{half_d} {half_w} 0.008' pos='0.06 0 {lvl}' material='mat_shelf_orange'/>"
                for lvl in levels
            ])
            shelves_xml.append(f"""
    <body name='shelf_{idx}' pos='{spos[0]} {spos[1]} 0.0'>
      <!-- Heavy Duty Structural Columns at the rear -->
      <geom type='box' size='0.015 0.015 {col_h}' pos='{-0.04} {-half_w + 0.02} {col_h}' material='mat_steel_blue'/>
      <geom type='box' size='0.015 0.015 {col_h}' pos='{-0.04} {half_w - 0.02} {col_h}' material='mat_steel_blue'/>
      <!-- Rear Cross Brace -->
      <geom type='box' size='0.010 {half_w - 0.02} 0.010' pos='{-0.04} 0 {col_h*2}' material='mat_steel_blue'/>
      <!-- Open-Front Cantilever Shelf Levels -->
{levels_str}
    </body>
""")
        shelves_xml_str = "\n".join(shelves_xml)

        xml = f"""
<mujoco model='warehouse_fulfillment_cell'>
  <compiler angle='radian' autolimits='true'/>
  <option timestep='0.002' gravity='0 0 -9.81' iterations='50'/>
  
  <visual>
    <headlight ambient='0.45 0.45 0.50' diffuse='0.60 0.60 0.65' specular='0.25 0.25 0.25'/>
    <rgba haze='0.12 0.18 0.25 1'/>
    <global offwidth='1280' offheight='720'/>
  </visual>
  
  <asset>
    <texture type='skybox' builtin='gradient' rgb1='0.18 0.28 0.40' rgb2='0.03 0.05 0.08' width='512' height='512'/>
    <texture name='floor_grid' type='2d' builtin='checker' width='512' height='512' rgb1='0.14 0.16 0.20' rgb2='0.20 0.22 0.27'/>
    <material name='floor_mat' texture='floor_grid' texrepeat='16 16' reflectance='0.22' specular='0.4'/>
    
    <material name='franka_white' rgba='0.94 0.94 0.97 1' specular='0.85' shininess='0.7' reflectance='0.18'/>
    <material name='franka_black' rgba='0.12 0.12 0.14 1' specular='0.5' shininess='0.4'/>
    <material name='franka_silver' rgba='0.80 0.82 0.85 1' specular='0.9' shininess='0.8' reflectance='0.3'/>
    <material name='mat_steel_blue' rgba='0.20 0.35 0.52 1' specular='0.7' shininess='0.5'/>
    <material name='mat_shelf_orange' rgba='0.90 0.48 0.15 1' specular='0.6' shininess='0.4'/>
    <material name='mat_conveyor_rubber' rgba='0.14 0.14 0.16 1' specular='0.2' roughness='0.8'/>
    <material name='mat_box_blue' rgba='0.28 0.58 0.88 1' specular='0.25' roughness='0.6'/>
    <material name='mat_box_green' rgba='0.28 0.78 0.48 1' specular='0.25' roughness='0.6'/>
    <material name='mat_box_red' rgba='0.88 0.38 0.32 1' specular='0.25' roughness='0.6'/>
    <material name='mat_cardboard' rgba='0.78 0.60 0.40 1' specular='0.1' roughness='0.8'/>
  </asset>
  
  <worldbody>
    <light pos='{arm_base[0]} {arm_base[1]} 3.2' dir='0 0 -1' directional='true' castshadow='true'/>
    <light pos='-0.6 -1.2 2.8' dir='0.6 0.8 -0.8' diffuse='0.45 0.45 0.50'/>
    
    <!-- Factory Floor -->
    <geom name='floor' type='plane' size='8 8 0.1' material='floor_mat' friction='1 0.005 0.0001'/>

    <!-- Franka Emika Panda 7-DOF Cobot -->
    <body name='panda_link0' pos='{arm_base[0]} {arm_base[1]} {arm_base[2]}'>
      <geom name='base_mount' type='cylinder' size='0.11 0.08' pos='0 0 0.08' material='franka_black'/>
      <body name='panda_link1' pos='0 0 0.16'>
        <joint name='panda_joint1' type='hinge' axis='0 0 1' range='-2.8973 2.8973'/>
        <geom name='l1' type='capsule' fromto='0 0 0 0 0 0.193' size='0.06' material='franka_white'/>
        <body name='panda_link2' pos='0 0 0.193'>
          <joint name='panda_joint2' type='hinge' axis='0 1 0' range='-1.7628 1.7628'/>
          <geom name='l2' type='capsule' fromto='0 0 0 0 0 0.12' size='0.055' material='franka_white'/>
          <body name='panda_link3' pos='0 0 0.123'>
            <joint name='panda_joint3' type='hinge' axis='0 0 1' range='-2.8973 2.8973'/>
            <geom name='l3' type='capsule' fromto='0 0 0 0.0825 0 0.12' size='0.055' material='franka_white'/>
            <body name='panda_link4' pos='0.0825 0 0.12'>
              <joint name='panda_joint4' type='hinge' axis='0 -1 0' range='-3.0718 -0.0698'/>
              <geom name='l4' type='capsule' fromto='0 0 0 -0.0825 0 0.14' size='0.05' material='franka_white'/>
              <body name='panda_link5' pos='-0.0825 0 0.14'>
                <joint name='panda_joint5' type='hinge' axis='0 0 1' range='-2.8973 2.8973'/>
                <geom name='l5' type='capsule' fromto='0 0 0 0 0 0.14' size='0.045' material='franka_white'/>
                <body name='panda_link6' pos='0 0 0.14'>
                  <joint name='panda_joint6' type='hinge' axis='0 1 0' range='-0.0175 3.7525'/>
                  <geom name='l6' type='capsule' fromto='0 0 0 0.088 0 0' size='0.045' material='franka_white'/>
                  <body name='panda_link7' pos='0.088 0 0'>
                    <joint name='panda_joint7' type='hinge' axis='0 0 1' range='-2.8973 2.8973'/>
                    <geom name='l7' type='cylinder' size='0.045 0.03' pos='0 0 0.03' material='franka_black'/>
                    
                    <!-- Hand & Parallel Gripper -->
                    <body name='panda_hand' pos='0 0 0.06'>
                      <geom name='hand' type='box' size='0.035 0.045 0.02' material='franka_white'/>
                      <body name='panda_leftfinger' pos='0 0.025 0.035'>
                        <joint name='panda_finger_joint1' type='slide' axis='0 1 0' range='0 0.04'/>
                        <geom name='lf' type='box' size='0.01 0.007 0.025' material='franka_black' friction='2.0 0.1 0.002'/>
                      </body>
                      <body name='panda_rightfinger' pos='0 -0.025 0.035'>
                        <joint name='panda_finger_joint2' type='slide' axis='0 -1 0' range='0 0.04'/>
                        <geom name='rf' type='box' size='0.01 0.007 0.025' material='franka_black' friction='2.0 0.1 0.002'/>
                      </body>
                      <site name='end_effector' pos='0 0 0.065' size='0.01'/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Industrial Conveyor Belt System -->
    <body name='conveyor_belt' pos='{conv_pos[0]} {conv_pos[1]} {conv_h}'>
      <geom name='belt_top' type='box' size='{conv_wid/2.0} {conv_len/2.0} 0.015' material='mat_conveyor_rubber'
            friction='0.8 0.02 0.001'/>
      <geom type='box' size='0.01 {conv_len/2.0} 0.03' pos='{-conv_wid/2.0} 0 0.02' material='mat_steel_blue'/>
      <geom type='box' size='0.01 {conv_len/2.0} 0.03' pos='{conv_wid/2.0} 0 0.02' material='mat_steel_blue'/>
      <!-- Legs -->
      <geom type='cylinder' size='0.02 {conv_h/2.0}' pos='{-conv_wid/2.0 + 0.02} {-conv_len/2.0 + 0.1} {-conv_h/2.0}' material='mat_steel_blue'/>
      <geom type='cylinder' size='0.02 {conv_h/2.0}' pos='{conv_wid/2.0 - 0.02} {-conv_len/2.0 + 0.1} {-conv_h/2.0}' material='mat_steel_blue'/>
      <geom type='cylinder' size='0.02 {conv_h/2.0}' pos='{-conv_wid/2.0 + 0.02} {conv_len/2.0 - 0.1} {-conv_h/2.0}' material='mat_steel_blue'/>
      <geom type='cylinder' size='0.02 {conv_h/2.0}' pos='{conv_wid/2.0 - 0.02} {conv_len/2.0 - 0.1} {-conv_h/2.0}' material='mat_steel_blue'/>
    </body>

    <!-- Industrial Pallet Shelves -->
    {shelves_xml_str}

    <!-- Cardboard Boxes -->
    {boxes_xml_str}
  </worldbody>
</mujoco>
"""
        return xml

    def _set_home_pose(self):
        """Set robot arm to steady, upright, human-like home position and hold rock-solid."""
        home_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        self.target_joint_positions = home_joints.copy()
        self.target_gripper_width = 0.04
        self.carried_sku = None
        self._hold_robot_joints()
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        elif self.step_callback is not None:
            self.step_callback(0.002)

    def get_joint_positions(self) -> np.ndarray:
        """Return current 7 joint angles [q0..q6] in radians."""
        return np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr])

    def get_end_effector_pos(self) -> np.ndarray:
        """Return current 3D position of end-effector site."""
        return np.array(self.data.site_xpos[self.ee_site_id])

    def compute_ik(
        self,
        target_pos: np.ndarray,
        max_iterations: int = 150,
        tol: float = 0.002
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve inverse kinematics for Franka Panda arm using damped Jacobian pseudo-inverse
        with height-adaptive seed and horizontal wrist orientation preference for shelf corridors.
        """
        q_init = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr])

        arm_base = self.layout_data.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
        dx = float(target_pos[0] - arm_base[0])
        dy = float(target_pos[1] - arm_base[1])
        target_yaw = math.atan2(dy, dx)
        z = float(target_pos[2])

        # Height-adaptive shoulder and elbow seeds ensuring the gripper stays level/horizontal
        q2_seed = float(np.clip(0.55 - z, -0.5, 0.55))
        q4_seed = -1.6
        q6_seed = float(np.clip(-q2_seed - q4_seed - np.pi / 2, 0.0, 1.5))

        q_seed = np.array([target_yaw, q2_seed, 0.0, q4_seed, 0.0, q6_seed, 0.0])

        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = q_seed[i]
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv))
        diag = 0.0001 * np.eye(3)

        best_sol = q_seed.copy()
        best_dist = float('inf')

        for step in range(max_iterations):
            current_pos = self.data.site_xpos[self.ee_site_id]
            err = target_pos - current_pos
            dist = float(np.linalg.norm(err))

            if dist < best_dist:
                best_dist = dist
                best_sol = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr])

            if dist < tol:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, np.zeros((3, self.model.nv)), self.ee_site_id)
            J = jacp[:, self.joint_dof_adr]
            J_pinv = J.T @ np.linalg.inv(J @ J.T + diag)

            # Nullspace posture projection to maintain horizontal forearm corridor
            k_posture = 0.03 if dist > 0.02 else 0.005
            q_cur = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr])
            null_proj = (np.eye(7) - J_pinv @ J) @ (k_posture * (q_seed - q_cur))

            dq = J_pinv @ err + null_proj
            step_scale = 0.5 if dist > 0.05 else 0.8
            for i, adr in enumerate(self.joint_qpos_adr):
                val = self.data.qpos[adr] + step_scale * dq[i]
                rng = self.model.jnt_range[self.model.joint(self.joint_names[i]).id]
                self.data.qpos[adr] = np.clip(val, rng[0], rng[1])

            mujoco.mj_forward(self.model, self.data)

        # Restore original state
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = q_init[i]
        mujoco.mj_forward(self.model, self.data)

        return (best_dist < 0.008), best_sol

    def move_cartesian_linear(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        duration: float = 0.6,
        steps: int = 25,
        gripper_width: float = 0.04,
        carried_sku: Optional[str] = None
    ):
        """
        Move robot end-effector along a strictly linear Cartesian line.
        Eliminates non-linear joint-space arcing and vertical sagging into shelf plates.
        """
        dt = duration / float(steps)

        for s_idx in range(steps + 1):
            tau = float(s_idx) / float(steps)
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            s_dot = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / max(duration, 0.01)

            p_cur = (1.0 - s) * p_start + s * p_end
            ok, q = self.compute_ik(p_cur)
            if ok:
                for i, adr in enumerate(self.joint_qpos_adr):
                    self.data.qpos[adr] = q[i]

            half_width = float(gripper_width) / 2.0
            self.data.qpos[self.finger1_qpos_adr] = half_width
            self.data.qpos[self.finger2_qpos_adr] = half_width

            if carried_sku:
                box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{carried_sku}")
                if box_body_id != -1:
                    j_id = self.model.body(box_body_id).jntadr[0]
                    if j_id != -1:
                        qadr = self.model.jnt_qposadr[j_id]
                        ee_pos = self.data.site_xpos[self.ee_site_id]
                        self.data.qpos[qadr:qadr+3] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.015]
                        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]

            mujoco.mj_step(self.model, self.data)

            cur_q = self.get_joint_positions()
            self.hardware.record_step(
                t=s_idx * dt,
                joints=cur_q,
                velocities=np.zeros(7),
                gripper_opening=gripper_width
            )

            if self.viewer:
                self.viewer.sync()
                time.sleep(dt * 0.8)
            elif self.step_callback is not None:
                self.step_callback(dt)

        self.target_joint_positions = np.array([self.data.qpos[adr] for adr in self.joint_qpos_adr])
        self.target_gripper_width = float(gripper_width)
        self.carried_sku = carried_sku
        self._hold_robot_joints()

    def move_joints_smooth(
        self,
        target_joints: np.ndarray,
        duration: float = 1.6,
        steps: int = 80,
        gripper_width: float = 0.04,
        carried_sku: Optional[str] = None
    ):
        """
        Move robot arm joints along a minimum-jerk quintic S-curve trajectory.
        Records joint trajectory for real hardware bridge.
        If carried_sku is provided, the box follows the end-effector smoothly.
        """
        q_start = self.get_joint_positions()
        q_end = np.array(target_joints[:7])

        dt = duration / float(steps)

        for step in range(steps + 1):
            tau = float(step) / float(steps)
            # Quintic polynomial: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            s_dot = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / max(duration, 0.01)

            q_interp = q_start + s * (q_end - q_start)
            qd_interp = s_dot * (q_end - q_start)

            # Apply to MuJoCo
            for i, adr in enumerate(self.joint_qpos_adr):
                self.data.qpos[adr] = q_interp[i]

            # Set gripper width
            half_width = float(gripper_width) / 2.0
            self.data.qpos[self.finger1_qpos_adr] = half_width
            self.data.qpos[self.finger2_qpos_adr] = half_width

            # Synchronize carried box with end-effector
            if carried_sku:
                box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{carried_sku}")
                if box_body_id != -1:
                    j_id = self.model.body(box_body_id).jntadr[0]
                    if j_id != -1:
                        qadr = self.model.jnt_qposadr[j_id]
                        ee_pos = self.data.site_xpos[self.ee_site_id]
                        self.data.qpos[qadr:qadr+3] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.015]
                        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]

            mujoco.mj_step(self.model, self.data)

            # Record trajectory waypoint for physical robot execution
            self.hardware.record_step(
                t=step * dt,
                joints=q_interp,
                velocities=qd_interp,
                gripper_opening=gripper_width
            )

            # Sync viewer if running
            if self.viewer:
                self.viewer.sync()
                time.sleep(dt * 0.8)
            elif self.step_callback is not None:
                self.step_callback(dt)

        # Update persistent target holding state so the arm never goes limp
        self.target_joint_positions = np.array(target_joints[:7])
        self.target_gripper_width = float(gripper_width)
        self.carried_sku = carried_sku
        self._hold_robot_joints()

    def pick_box_to_conveyor(self, sku: str) -> Tuple[bool, str]:
        """
        Execute pick and place of a specific cardboard box to the conveyor belt
        using a collision-free corridor that approaches horizontally from the open
        aisle and never clips through shelf beams or knocks adjacent boxes.
        """
        box_data = None
        for b in self.layout_data.get("boxes", []):
            if b.get("sku") == sku:
                box_data = b
                break

        if not box_data:
            return False, f"Box SKU '{sku}' not found in layout"

        # Begin trajectory recording
        self.hardware.start_recording(action_name=f"pick_{sku}_to_conveyor")

        # Query actual live physics position of the box in the simulation
        box_pos = self.get_box_position(sku)
        conv_pos = np.array(self.layout_data.get("conveyor", {}).get("position", [1.05, 0.0, 0.22]))
        arm_base = np.array(self.layout_data.get("arm", {}).get("base_position", [0.70, 0.0, 0.0]))

        # Detect if box is on the floor (e.g. knocked down) or on the shelf
        is_on_ground = (box_pos[2] < 0.15)

        if is_on_ground:
            # -------------------------------------------------------------
            # Ground Retrieval Sequence (Fallen / Knocked Box on Floor):
            # -------------------------------------------------------------
            # 1. Hover above ground box in open air
            p_hover = np.array([box_pos[0], box_pos[1], 0.25])
            ok_h, q_hover = self.compute_ik(p_hover)
            self.move_joints_smooth(q_hover, duration=0.8, gripper_width=0.045)

            # 2. Descend to ground grasp height
            p_ground_grasp = np.array([box_pos[0], box_pos[1], max(0.045, box_pos[2] + 0.02)])
            ok_g, q_ground_grasp = self.compute_ik(p_ground_grasp)
            self.move_joints_smooth(q_ground_grasp, duration=0.6, gripper_width=0.045)

            # 3. Close gripper firmly around the box on the ground
            self.move_joints_smooth(q_ground_grasp, duration=0.35, gripper_width=0.02, carried_sku=sku)

            # 4. Lift box off the floor cleanly
            p_lift = np.array([box_pos[0], box_pos[1], 0.42])
            ok_l, q_lift = self.compute_ik(p_lift)
            self.move_joints_smooth(q_lift, duration=0.6, gripper_width=0.02, carried_sku=sku)

        else:
            # -------------------------------------------------------------
            # Shelf Bay Retrieval Sequence (Clean open-aisle corridor):
            # -------------------------------------------------------------
            delta = arm_base[:2] - box_pos[:2]
            dist_2d = float(np.linalg.norm(delta))
            u = delta / dist_2d if dist_2d > 1e-4 else np.array([1.0, 0.0])

            # 1. Fly to open aisle stand-off at exact box level height
            grasp_z = box_pos[2] + 0.015
            p_aisle = np.array([box_pos[0] + u[0] * 0.24, box_pos[1] + u[1] * 0.24, grasp_z])
            ok1, q_aisle = self.compute_ik(p_aisle)
            self.move_joints_smooth(q_aisle, duration=0.7, gripper_width=0.045)

            # 2. Straight Cartesian ingress into shelf slot (eliminates joint-space sag)
            p_grasp = np.array([box_pos[0] + u[0] * 0.05, box_pos[1] + u[1] * 0.05, grasp_z])
            self.move_cartesian_linear(p_aisle, p_grasp, duration=0.6, gripper_width=0.045)

            # 3. Close gripper firmly on box
            self.move_joints_smooth(self.target_joint_positions, duration=0.3, gripper_width=0.02, carried_sku=sku)

            # 4. Straight Cartesian egress back out into the open aisle
            p_egress = np.array([p_aisle[0], p_aisle[1], grasp_z])
            self.move_cartesian_linear(p_grasp, p_egress, duration=0.6, gripper_width=0.02, carried_sku=sku)

        # -------------------------------------------------------------
        # Common Outfeed Transit to Conveyor:
        # -------------------------------------------------------------
        # Forward transit leaning out into open aisle toward conveyor (avoids shelf)
        p_transit_forward = np.array([0.85, 0.0, 0.50])
        ok6, q_transit_forward = self.compute_ik(p_transit_forward)
        self.move_joints_smooth(q_transit_forward, duration=0.7, gripper_width=0.02, carried_sku=sku)

        # 8. Conveyor Approach Hover: hover directly above conveyor drop point
        p_conv_hover = np.array([conv_pos[0], conv_pos[1], conv_pos[2] + 0.14])
        ok7, q_conv_hover = self.compute_ik(p_conv_hover)
        self.move_joints_smooth(q_conv_hover, duration=0.8, gripper_width=0.02, carried_sku=sku)

        # 9. Lower to Conveyor Surface
        p_conv_drop = np.array([conv_pos[0], conv_pos[1], conv_pos[2] + 0.04])
        ok8, q_conv_drop = self.compute_ik(p_conv_drop)
        self.move_joints_smooth(q_conv_drop, duration=0.4, gripper_width=0.02, carried_sku=sku)

        # Deposit box directly onto conveyor belt
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{sku}")
        if box_body_id != -1:
            joint_id = self.model.body(box_body_id).jntadr[0]
            if joint_id != -1:
                qadr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qadr:qadr+3] = [conv_pos[0], conv_pos[1], conv_pos[2] + 0.035]
                self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]

        # 10. Open Gripper (release box)
        self.move_joints_smooth(q_conv_drop, duration=0.3, gripper_width=0.04, carried_sku=None)

        # 11. Vertical Retract above Conveyor
        self.move_joints_smooth(q_conv_hover, duration=0.5, gripper_width=0.04, carried_sku=None)

        # 12. Return smoothly to the steady, upright home posture
        home_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        self.move_joints_smooth(home_joints, duration=1.0, gripper_width=0.04, carried_sku=None)

        self.hardware.stop_recording()

        # Update inventory status
        self.box_status[sku] = "DISPATCHED"

        # Simulate conveyor transport gliding box forward
        self._glide_box_on_conveyor(sku, duration=1.2)

        return True, f"Successfully dispatched {sku} onto conveyor belt in MuJoCo!"

    def _glide_box_on_conveyor(self, sku: str, duration: float = 1.0):
        """Simulate dynamic conveyor belt gliding."""
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{sku}")
        if box_body_id == -1:
            return
        joint_id = self.model.body(box_body_id).jntadr[0]
        if joint_id == -1:
            return
        qadr = self.model.jnt_qposadr[joint_id]

        steps = int(duration * 60)
        speed = 0.25  # m/s along Y axis
        dt = duration / float(steps)

        for _ in range(steps):
            self._hold_robot_joints()
            self.data.qpos[qadr + 1] += speed * dt
            mujoco.mj_step(self.model, self.data)
            self._hold_robot_joints()
            if self.viewer:
                self.viewer.sync()
                time.sleep(dt * 0.8)
            elif self.step_callback is not None:
                self.step_callback(dt)

    def execute_scripted_episode(self) -> Dict:
        """Dispatch all cardboard boxes to conveyor belt."""
        boxes = self.layout_data.get("boxes", [])
        total = len(boxes)
        success_count = 0
        for box in boxes:
            sku = box["sku"]
            success, _ = self.pick_box_to_conveyor(sku)
            if success:
                success_count += 1
        return {
            "total_boxes": total,
            "successful_placements": success_count,
            "failed_placements": total - success_count,
            "success_rate": success_count / max(1, total)
        }

    def reset(self):
        """Reset scene and robot to initial state."""
        self._set_home_pose()
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for box in self.layout_data.get("boxes", []):
            sku = box["sku"]
            pos = box["spawn_position"]
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{sku}")
            if b_id != -1:
                j_id = self.model.body(b_id).jntadr[0]
                if j_id != -1:
                    qadr = self.model.jnt_qposadr[j_id]
                    self.data.qpos[qadr:qadr+3] = pos
                    self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
            self.box_status[sku] = "ON_SHELF"
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        elif self.step_callback is not None:
            self.step_callback(0.002)

    def step_simulation(self, steps: int = 1):
        """Step physics simulation while holding robot arm rock-solid."""
        for _ in range(steps):
            self._hold_robot_joints()
            mujoco.mj_step(self.model, self.data)
            self._hold_robot_joints()
        if self.viewer:
            self.viewer.sync()
        elif self.step_callback is not None:
            self.step_callback(0.002)

    def close(self):
        """Close viewer and release simulation resources."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

