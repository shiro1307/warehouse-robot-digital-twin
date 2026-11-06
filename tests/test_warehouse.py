"""
Unit and integration tests for the Warehouse Sorting Simulation (MuJoCo Physics Engine).
"""

import unittest
import os
import numpy as np

from warehouse_sim import config
from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv


class TestWarehouseSimulation(unittest.TestCase):
    """Test suite for warehouse simulation components and motion primitives using MuJoCo."""

    def setUp(self):
        from warehouse_sim.layout_manager import LayoutManager
        LayoutManager.save_layout(LayoutManager.get_preset("standard"), "layout.json")
        self.env = MuJoCoWarehouseEnv(gui=False, layout_file="layout.json")

    def tearDown(self):
        self.env.close()

    def test_environment_reset_and_state(self):
        """Verify environment reset produces expected state observation structure."""
        self.env.reset()
        self.assertEqual(len(self.env.box_status), 6)
        joints = self.env.get_joint_positions()
        self.assertEqual(len(joints), 7)
        ee_pos = self.env.get_end_effector_pos()
        self.assertEqual(len(ee_pos), 3)

    def test_conveyor_pick_and_dispatch(self):
        """Verify picking a chosen cardboard box and dispatching it to the conveyor belt."""
        self.env.reset()
        success, msg = self.env.pick_box_to_conveyor("ELEC-101")
        self.assertTrue(success, f"Conveyor dispatch failed: {msg}")
        self.assertEqual(self.env.box_status["ELEC-101"], "DISPATCHED")

    def test_ik_solver_reachability(self):
        """Verify forward-kinematics verified IK solver finds accurate joint solutions."""
        for i, box in enumerate(self.env.layout_data.get("boxes", [])):
            pos = box["spawn_position"]
            ok, ik_sol = self.env.compute_ik(pos)
            self.assertTrue(ok, f"IK failed for box {box['sku']} at position {pos}")
            self.assertEqual(len(ik_sol), 7)

    def test_scripted_episode_execution(self):
        """Verify complete scripted pick-and-place episode completes with 100% success rate."""
        self.env.reset()
        stats = self.env.execute_scripted_episode()
        self.assertEqual(stats['total_boxes'], 6)
        self.assertEqual(stats['successful_placements'], 6)
        self.assertEqual(stats['failed_placements'], 0)
        self.assertAlmostEqual(stats['success_rate'], 1.0, places=2)

    def test_layout_manager_and_presets(self):
        """Verify workcell layout manager and presets are kinematically reachable."""
        from warehouse_sim.layout_manager import LayoutManager
        for preset_name in ["standard", "side_conveyor", "compact"]:
            preset = LayoutManager.get_preset(preset_name)
            is_valid, warnings = LayoutManager.validate(preset)
            self.assertTrue(is_valid, f"Preset '{preset_name}' has reachability warnings: {warnings}")

    def test_hardware_bridge_trajectory_recording(self):
        """Verify hardware bridge captures joint waypoints for real Franka arm deployment."""
        from warehouse_sim.hardware_bridge import HardwareBridge
        bridge = HardwareBridge()
        bridge.start_recording()
        # Record sample waypoints
        bridge.record_step(
            t=0.0,
            joints=np.zeros(7),
            velocities=np.zeros(7),
            gripper_opening=0.04
        )
        bridge.record_step(
            t=0.1,
            joints=np.array([0.1, -0.2, 0.0, -1.5, 0.0, 1.2, 0.785]),
            velocities=np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper_opening=0.0
        )
        bridge.stop_recording()
        self.assertEqual(len(bridge.recorded_trajectories), 1)
        self.assertEqual(len(bridge.recorded_trajectories[0]["waypoints"]), 2)
        
        # Test export
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_json = f.name
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_csv = f.name
            
        try:
            bridge.export_trajectory_json(temp_json)
            bridge.export_trajectory_csv(temp_csv)
            self.assertTrue(os.path.getsize(temp_json) > 100)
            self.assertTrue(os.path.getsize(temp_csv) > 50)
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    def test_desktop_app_initialization(self):
        """Verify the desktop app UI initializes tabs and widgets properly."""
        import tkinter as tk
        from warehouse_sim.app import RoboWarehouseApp
        root = tk.Tk()
        # Prevent root from displaying during headless test
        root.withdraw()
        app = RoboWarehouseApp(root)
        self.assertIsNotNone(app.canvas)
        self.assertIsNotNone(app.txt_log)
        self.assertEqual(len(app.notebook.tabs()), 4)
        app._on_close()

    def test_mujoco_environment_dispatch(self):
        """Verify MuJoCo warehouse environment compiles and executes pick action."""
        from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv
        m_env = MuJoCoWarehouseEnv(gui=False, layout_file="layout.json")
        self.assertEqual(len(m_env.box_status), 6)
        ok, msg = m_env.pick_box_to_conveyor("ELEC-101")
        self.assertTrue(ok, f"MuJoCo pick failed: {msg}")
        self.assertEqual(m_env.box_status["ELEC-101"], "DISPATCHED")
        self.assertTrue(len(m_env.hardware.recorded_trajectories) >= 1)
        m_env.close()

    def test_builder_drag_and_synchronization(self):
        """Verify builder shelf and box synchronization mechanics."""
        import tkinter as tk
        from warehouse_sim.app import RoboWarehouseApp
        root = tk.Tk()
        root.withdraw()
        app = RoboWarehouseApp(root)

        # Check initial shelf mapping
        self.assertTrue(len(app.box_shelf_map) >= 6)
        initial_shelf0_pos = list(app.current_layout["shelves"][0]["position"])
        box0_sku = app.current_layout["boxes"][0]["sku"]
        box0_init_pos = list(app.current_layout["boxes"][0]["spawn_position"])

        # Nudge shelf 0 by +0.06m in X
        app.selected_item = "shelf_0"
        app._nudge_selected(0.06, 0.0)
        new_shelf0_pos = app.current_layout["shelves"][0]["position"]
        self.assertAlmostEqual(new_shelf0_pos[0], initial_shelf0_pos[0] + 0.06, places=2)

        # If box 0 was on shelf 0, verify it moved with the shelf
        if app.box_shelf_map.get(box0_sku) == 0:
            box0_new_pos = app.current_layout["boxes"][0]["spawn_position"]
            self.assertAlmostEqual(box0_new_pos[0], box0_init_pos[0] + 0.06, places=2)

        # Test preset loading
        app._apply_preset("compact")
        self.assertEqual(len(app.current_layout["shelves"]), 1)
        self.assertEqual(len(app.current_layout["boxes"]), 6)
        app._apply_preset("standard")
        app._on_close()

    def test_builder_apply_layout_to_mujoco(self):
        """Verify MuJoCo loads in-memory layout data dynamically."""
        from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv
        from warehouse_sim.layout_manager import LayoutManager
        import copy

        temp_layout_path = "test_temp_layout.json"
        try:
            custom_layout = copy.deepcopy(LayoutManager.get_preset("side_conveyor"))
            custom_layout["arm"]["base_position"] = [0.55, 0.05, 0.0]

            env = MuJoCoWarehouseEnv(gui=False, layout_file=temp_layout_path, layout_data=custom_layout)
            # Verify arm link0 base position in MuJoCo model matches custom layout
            link0_id = env.model.body("panda_link0").id
            base_pos = env.model.body_pos[link0_id]
            self.assertAlmostEqual(base_pos[0], 0.55, places=2)
            self.assertAlmostEqual(base_pos[1], 0.05, places=2)

            # Verify pick-and-place works in the custom layout
            ok, msg = env.pick_box_to_conveyor("ELEC-101")
            self.assertTrue(ok, f"Custom layout pick failed: {msg}")
            env.close()
        finally:
            if os.path.exists(temp_layout_path):
                os.remove(temp_layout_path)

    def test_ground_pick_and_zero_knock(self):
        """Verify picking boxes knocked to the ground and collision-free shelf picks."""
        from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv
        import mujoco

        env = MuJoCoWarehouseEnv(gui=False, layout_file="layout.json")

        # 1. Knock TOOL-202 to the ground and verify ground retrieval without teleportation
        box_id = env.model.body("box_TOOL-202").id
        qadr = env.model.jnt_qposadr[env.model.body(box_id).jntadr[0]]
        env.data.qpos[qadr:qadr+3] = [0.65, -0.28, 0.035]
        mujoco.mj_forward(env.model, env.data)

        ok, msg = env.pick_box_to_conveyor("TOOL-202")
        self.assertTrue(ok, f"Ground pick failed: {msg}")
        final_pos = env.get_box_position("TOOL-202")
        self.assertGreater(final_pos[2], 0.20)
        self.assertEqual(env.box_status["TOOL-202"], "DISPATCHED")

        # 2. Reset and pick shelf box, verifying zero knocked boxes
        env.reset()
        ok_shelf, msg_shelf = env.pick_box_to_conveyor("PHAR-505")
        self.assertTrue(ok_shelf, f"Shelf pick failed: {msg_shelf}")
        for b in env.layout_data["boxes"]:
            if b["sku"] == "PHAR-505":
                continue
            cur_p = env.get_box_position(b["sku"])
            orig_p = np.array(b["spawn_position"])
            self.assertLess(np.linalg.norm(cur_p - orig_p), 0.04)

        env.close()


if __name__ == '__main__':
    unittest.main()

