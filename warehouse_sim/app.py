"""
RoboWarehouse Studio & Game Application.

A native desktop app providing:
1. Warehouse Fulfillment Game: Select cardboard boxes from industrial shelf
   to place on conveyor belt with scoring, sound/visual feedback, and stats.
2. Interactive 2D Workcell Builder: Click and drag robot arm, conveyor, and
   shelves anywhere on the factory floor with real-time kinematic reach checking.
3. 3D Digital Twin Viewer: Real-time MuJoCo PBR 3D visualization.
4. Physical Robot Hardware Deployment: Direct bridge to physical Franka Panda arm
   via FCI / frankapy.
"""

import sys
import os

# Configure High-Performance Discrete Dedicated GPU (NVIDIA RTX 2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SHIM_MCCOMPAT"] = "0x800000001"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

if sys.platform == "win32":
    try:
        import ctypes
        _dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpu_enable.dll")
        if os.path.exists(_dll_path):
            ctypes.CDLL(_dll_path)
    except Exception:
        pass

import time
import math
import json
import queue
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from warehouse_sim import config
from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv
from warehouse_sim.layout_manager import LayoutManager
from warehouse_sim.hardware_bridge import HardwareBridge


# ---------------------------------------------------------------------------
# Background Simulation Engine Thread (MuJoCo Physics)
# ---------------------------------------------------------------------------

class SimEngine:
    """
    Background simulation worker that manages MuJoCo in a dedicated thread,
    keeping the Tkinter UI 100% responsive.
    Supports offscreen high-speed rendering streamed directly to Tkinter via PPM byte stream,
    or pop-out native GLFW OpenGL window.
    """

    def __init__(self, ui_queue: queue.Queue, render_embedded: bool = True):
        self.ui_queue = ui_queue
        self.cmd_queue = queue.Queue()
        self.render_embedded = render_embedded
        self.render_w = 580
        self.render_h = 440
        self.env = None
        self.renderer = None
        self.cam = None
        self.opt = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.is_busy = False
        self.current_layout_file = "layout.json"
        self.current_layout_data = None
        self.last_render_time = 0.0

    def start(self, layout_file: str = "layout.json"):
        """Start the background simulation engine thread."""
        if self.running:
            return
        self.current_layout_file = layout_file
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the background simulation engine."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.cmd_queue.put(("STOP", None))
            self.thread.join(timeout=2.0)

    def send_command(self, cmd: str, data=None):
        """Send a thread-safe command to the simulation engine."""
        self.cmd_queue.put((cmd, data))

    def _init_env(self):
        """Instantiate environment with either embedded offscreen or native GUI viewer."""
        if not self.running:
            return
        try:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None
            self.renderer = None

            import mujoco
            mode_str = "Embedded Viewport" if self.render_embedded else "Pop-Out Window"
            self.ui_queue.put(("STATUS", f"Starting MuJoCo 3D Engine ({mode_str})..."))
            self.env = MuJoCoWarehouseEnv(
                gui=not self.render_embedded,
                layout_file=self.current_layout_file,
                layout_data=self.current_layout_data
            )
            if not self.running:
                if self.env is not None:
                    self.env.close()
                    self.env = None
                return

            if self.render_embedded:
                self.renderer = mujoco.Renderer(self.env.model, height=self.render_h, width=self.render_w)
                self.cam = mujoco.MjvCamera()
                self.opt = mujoco.MjvOption()
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.cam.distance = 2.4
                self.cam.azimuth = 135.0
                self.cam.elevation = -22.0
                self.cam.lookat = np.array([0.55, 0.0, 0.35])
                self.env.step_callback = self._on_env_step

                # Query GPU Hardware Details from OpenGL
                gpu_str = "NVIDIA GeForce RTX 2050"
                try:
                    from OpenGL import GL
                    r_name = GL.glGetString(GL.GL_RENDERER)
                    if r_name:
                        gpu_str = r_name.decode("utf-8", errors="ignore")
                except Exception:
                    pass
                self.ui_queue.put(("GPU_DETECTED", gpu_str))

            self.ui_queue.put(("STATUS", "MuJoCo 3D Simulation Ready"))
            self.ui_queue.put(("SIM_READY", self._get_env_snapshot()))
            self.ui_queue.put(("VIEW_MODE_CHANGED", self.render_embedded))
            self._render_frame()

        except Exception as e:
            self.ui_queue.put(("ERROR", f"Failed to initialize simulation: {e}"))
            self.running = False

    def _on_env_step(self, dt: float):
        """
        Callback invoked during robot trajectory steps for live smooth rendering at max GPU FPS.
        Drains camera commands continuously so user can freely orbit, pan, and zoom the camera
        while the arm is moving and boxes are being shifted.
        """
        has_cam_change = False
        while True:
            try:
                cmd, data = self.cmd_queue.get_nowait()
                if cmd in ("ORBIT_CAM", "PAN_CAM", "ZOOM_CAM", "SET_CAMERA"):
                    self._handle_command(cmd, data)
                    has_cam_change = True
                elif cmd == "STOP":
                    self.running = False
            except queue.Empty:
                break

        now = time.time()
        if has_cam_change or (now - self.last_render_time >= 0.016):
            self.last_render_time = now
            self._render_frame()
        time.sleep(min(dt * 0.5, 0.008))

    def _render_frame(self):
        """Render offscreen frame and post PPM binary buffer to UI queue."""
        if not self.render_embedded or self.env is None:
            return
        try:
            if self.renderer is not None:
                self.renderer.update_scene(self.env.data, camera=self.cam, scene_option=self.opt)
                pixels = self.renderer.render()
                ppm = f"P6 {self.render_w} {self.render_h} 255\n".encode("ascii") + pixels.tobytes()
                self.ui_queue.put(("FRAME", ppm))
        except Exception:
            pass

    def _run(self):
        """Main simulation thread loop."""
        self._init_env()
        if not self.running:
            return

        last_telemetry_time = 0.0

        while self.running:
            # Process all pending commands immediately without accumulating lag
            has_camera_change = False
            while True:
                try:
                    cmd, data = self.cmd_queue.get_nowait()
                    self._handle_command(cmd, data)
                    if cmd in ("ORBIT_CAM", "PAN_CAM", "ZOOM_CAM", "SET_CAMERA"):
                        has_camera_change = True
                except queue.Empty:
                    break

            if has_camera_change and self.render_embedded and not self.is_busy:
                self.last_render_time = time.time()
                self._render_frame()

            # Step conveyor physics and boxes continuously when idle
            if self.env is not None and not self.is_busy:
                try:
                    self.env.step_simulation(1)
                except Exception:
                    pass

                # Render idle viewport frames at maximum GPU refresh rate (60+ FPS)
                now = time.time()
                if self.render_embedded and (now - self.last_render_time >= 0.016):
                    self.last_render_time = now
                    self._render_frame()

            # Send telemetry to UI at 15 Hz
            now = time.time()
            if now - last_telemetry_time > 0.066 and self.env is not None:
                last_telemetry_time = now
                self._send_telemetry()

            time.sleep(0.004)

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        self.renderer = None

    def _handle_command(self, cmd: str, data):
        """Execute action requested by the UI."""
        if cmd == "STOP":
            self.running = False

        elif cmd == "PICK_BOX":
            box_id = data
            self.is_busy = True
            self.ui_queue.put(("ACTION_START", f"Picking box {box_id}..."))
            try:
                success, msg = self.env.pick_box_to_conveyor(box_id)
                self.ui_queue.put(("ACTION_COMPLETE", {
                    "success": success,
                    "message": msg,
                    "box_id": box_id,
                    "snapshot": self._get_env_snapshot()
                }))
            except Exception as e:
                self.ui_queue.put(("ACTION_COMPLETE", {
                    "success": False,
                    "message": f"Pick failed: {e}",
                    "box_id": box_id,
                    "snapshot": self._get_env_snapshot()
                }))
            finally:
                self.is_busy = False

        elif cmd == "DISPATCH_ALL":
            self.is_busy = True
            self.ui_queue.put(("ACTION_START", "Dispatching all remaining shelf boxes..."))
            try:
                stats = self.env.execute_scripted_episode()
                self.ui_queue.put(("ALL_COMPLETE", {
                    "stats": stats,
                    "snapshot": self._get_env_snapshot()
                }))
            except Exception as e:
                self.ui_queue.put(("ACTION_COMPLETE", {
                    "success": False,
                    "message": f"Episode failed: {e}",
                    "snapshot": self._get_env_snapshot()
                }))
            finally:
                self.is_busy = False

        elif cmd == "RESET_SCENE":
            self.is_busy = True
            self.ui_queue.put(("ACTION_START", "Restocking warehouse shelf boxes..."))
            try:
                if hasattr(self.env, "reset"):
                    self.env.reset()
                self.ui_queue.put(("RESET_COMPLETE", self._get_env_snapshot()))
                self._render_frame()
            except Exception as e:
                self.ui_queue.put(("ERROR", f"Reset failed: {e}"))
            finally:
                self.is_busy = False

        elif cmd == "RELOAD_LAYOUT":
            layout_dict = data
            self.is_busy = True
            self.current_layout_data = layout_dict
            self.ui_queue.put(("ACTION_START", "Applying new workcell layout to 3D world..."))
            try:
                # Save to layout.json on disk so changes persist
                LayoutManager.save_layout(layout_dict, self.current_layout_file)
                config.apply_layout(layout_dict)
                self._init_env()
                self.ui_queue.put(("LAYOUT_APPLIED", self._get_env_snapshot()))
            except Exception as e:
                self.ui_queue.put(("ERROR", f"Failed applying layout: {e}"))
            finally:
                self.is_busy = False

        elif cmd == "ORBIT_CAM":
            d_az, d_el = data
            if self.cam is not None:
                self.cam.azimuth = (self.cam.azimuth + d_az) % 360.0
                self.cam.elevation = max(-89.0, min(89.0, self.cam.elevation + d_el))

        elif cmd == "PAN_CAM":
            d_x, d_y = data
            if self.cam is not None:
                rad = math.radians(self.cam.azimuth)
                right = np.array([math.cos(rad), math.sin(rad), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                self.cam.lookat += right * d_x + up * d_y

        elif cmd == "ZOOM_CAM":
            d_dist = data
            if self.cam is not None:
                self.cam.distance = max(0.6, min(6.0, self.cam.distance + d_dist))

        elif cmd == "TOGGLE_VIEW_MODE":
            self.render_embedded = not self.render_embedded
            self._init_env()

        elif cmd == "SET_CAMERA":
            preset = data
            if self.cam is not None:
                if preset == "isometric":
                    self.cam.distance = 2.4
                    self.cam.azimuth = 135.0
                    self.cam.elevation = -22.0
                    self.cam.lookat = np.array([0.55, 0.0, 0.35])
                elif preset == "top_down":
                    self.cam.distance = 2.2
                    self.cam.azimuth = 90.0
                    self.cam.elevation = -89.0
                    self.cam.lookat = np.array([0.55, 0.0, 0.35])
                elif preset == "conveyor":
                    self.cam.distance = 1.4
                    self.cam.azimuth = 180.0
                    self.cam.elevation = -18.0
                    self.cam.lookat = np.array([0.55, 0.35, 0.25])
                elif preset == "front":
                    self.cam.distance = 2.1
                    self.cam.azimuth = 90.0
                    self.cam.elevation = -15.0
                    self.cam.lookat = np.array([0.55, 0.0, 0.35])

            if self.env is not None and hasattr(self.env, "viewer") and self.env.viewer:
                cam = self.env.viewer.cam
                if preset == "isometric":
                    cam.distance = 2.4
                    cam.azimuth = 135
                    cam.elevation = -22
                    cam.lookat = [0.55, 0.0, 0.35]
                elif preset == "top_down":
                    cam.distance = 2.2
                    cam.azimuth = 90
                    cam.elevation = -89
                    cam.lookat = [0.55, 0.0, 0.35]
                elif preset == "conveyor":
                    cam.distance = 1.4
                    cam.azimuth = 180
                    cam.elevation = -18
                    cam.lookat = [0.55, 0.35, 0.25]
                elif preset == "front":
                    cam.distance = 2.1
                    cam.azimuth = 90
                    cam.elevation = -15
                    cam.lookat = [0.55, 0.0, 0.35]

    def _get_env_snapshot(self) -> Dict:
        """Capture summary state of boxes and objects in MuJoCo."""
        if self.env is None:
            return {}
        boxes = []
        for b in self.env.layout_data.get("boxes", []):
            sku = b["sku"]
            is_on_conveyor = (self.env.box_status.get(sku) == "DISPATCHED")
            cur_pos = self.env.get_box_position(sku) if hasattr(self.env, "get_box_position") else b.get("spawn_position", [0, 0, 0])
            is_on_ground = (cur_pos[2] < 0.15) and not is_on_conveyor
            boxes.append({
                "body_id": sku,
                "sku": sku,
                "name": b["name"],
                "category": b.get("category", 0),
                "on_conveyor": is_on_conveyor,
                "on_ground": is_on_ground,
                "position": [round(float(v), 3) for v in cur_pos],
                "on_shelf": not is_on_conveyor and not is_on_ground
            })
        return {
            "boxes": boxes,
            "total_boxes": len(boxes),
            "boxes_on_shelf": sum(1 for b in boxes if b["on_shelf"]),
            "boxes_on_conveyor": sum(1 for b in boxes if b["on_conveyor"])
        }

    def _send_telemetry(self):
        """Send joint angles and end effector position to UI."""
        if self.env is None:
            return
        try:
            if hasattr(self.env, "get_joint_positions"):
                joints = self.env.get_joint_positions()
                ee_pos = self.env.get_end_effector_pos()
                self.ui_queue.put(("TELEMETRY", {
                    "joints": [round(float(j), 3) for j in joints[:7]],
                    "ee_pos": [round(float(p), 3) for p in ee_pos],
                    "is_open": True,
                    "is_closed": False,
                    "is_busy": self.is_busy
                }))
            else:
                joints = self.env.robot.get_joint_positions()
                ee_pos, ee_quat = self.env.robot.get_end_effector_pose()
                grip = self.env.robot.get_gripper_state()
                self.ui_queue.put(("TELEMETRY", {
                    "joints": [round(float(j), 3) for j in joints[:7]],
                    "ee_pos": [round(float(p), 3) for p in ee_pos],
                    "is_open": grip.get("is_open", True),
                    "is_closed": grip.get("is_closed", False),
                    "is_busy": self.is_busy
                }))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main Desktop App & Game UI
# ---------------------------------------------------------------------------

class RoboWarehouseApp:
    """
    Main desktop game and workcell studio window.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RoboWarehouse Studio - Franka Digital Twin & Game")
        self.root.geometry("1360x840")
        self.root.minsize(1080, 680)

        # Game State
        self.score = 0
        self.orders_fulfilled = 0
        self.combo = 1
        self.box_data: List[Dict] = []
        self.box_widgets: Dict[int, Dict] = {}

        # Workcell Builder State
        self.layout_mgr = LayoutManager()
        self.current_layout = self.layout_mgr.load_layout()
        self.canvas_scale = 220.0  # pixels per meter
        self.canvas_origin_x = 280
        self.canvas_origin_y = 230
        self.drag_item = None
        self.selected_item = "arm"
        self._is_updating_sliders = False
        self.box_shelf_map: Dict[str, int] = {}
        self.box_shelf_offsets: Dict[str, Tuple[float, float]] = {}
        self._map_boxes_to_shelves()

        # Viewport Mouse Interaction State
        self._vp_is_orbiting = False
        self._vp_is_panning = False
        self._vp_drag_start_x = None
        self._vp_drag_start_y = None
        self._vp_pan_start_x = None
        self._vp_pan_start_y = None
        self.sim_photo = None

        # Rolling Live FPS & GPU Stats
        self._fps_count = 0
        self._fps_last_time = time.time()
        self._current_fps = 60.0
        self.gpu_name = "NVIDIA GeForce RTX 2050"

        # UI Communication Queue
        self.ui_queue = queue.Queue()
        self.sim = SimEngine(self.ui_queue, render_embedded=True)

        # Setup Styles & UI Layout
        self._setup_theme()
        self._build_ui()

        # Start background simulation engine
        self.sim.start()

        # Start UI event polling loop at 60 Hz
        self.root.after(16, self._poll_engine)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_theme(self):
        """Configure modern dark industrial color scheme."""
        self.colors = {
            "bg": "#0b0e14",
            "sidebar": "#121620",
            "card": "#1a202c",
            "card_hover": "#222a3a",
            "card_border": "#283347",
            "accent": "#2563eb",
            "accent_light": "#38bdf8",
            "accent_hover": "#1d4ed8",
            "green": "#10b981",
            "green_bg": "#064e3b",
            "amber": "#f59e0b",
            "amber_bg": "#78350f",
            "red": "#ef4444",
            "red_bg": "#7f1d1d",
            "purple": "#8b5cf6",
            "purple_hover": "#7c3aed",
            "text": "#f8fafc",
            "text_muted": "#94a3b8",
            "text_dim": "#64748b",
            "grid": "#151a26",
            "grid_primary": "#20283b"
        }
        self.root.configure(bg=self.colors["bg"])

        style = ttk.Style()
        style.theme_use("clam")

        # Configure tabs
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=self.colors["sidebar"],
                        foreground=self.colors["text_muted"],
                        padding=[18, 9],
                        font=("Segoe UI", 9, "bold"),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", self.colors["card"])],
                  foreground=[("selected", self.colors["accent_light"])])

        style.configure("TProgressbar",
                        background=self.colors["accent_light"],
                        troughcolor=self.colors["sidebar"],
                        borderwidth=0)

    def _build_ui(self):
        """Construct application layout with header, split viewport + studio pane, and status bar."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=self.colors["sidebar"], height=60, padx=18, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        title_frame = tk.Frame(header, bg=self.colors["sidebar"])
        title_frame.pack(side=tk.LEFT)

        logo_lbl = tk.Label(title_frame, text="⚡", font=("Segoe UI", 15, "bold"),
                            bg=self.colors["sidebar"], fg=self.colors["accent_light"])
        logo_lbl.pack(side=tk.LEFT, padx=(0, 6))

        title_lbl = tk.Label(title_frame,
                             text="ROBOWAREHOUSE",
                             font=("Segoe UI", 13, "bold"),
                             bg=self.colors["sidebar"],
                             fg="#ffffff")
        title_lbl.pack(side=tk.LEFT)

        badge_pro = tk.Label(title_frame, text=" PRO v2.4 ", font=("Segoe UI", 8, "bold"),
                             bg=self.colors["accent"], fg="#ffffff", padx=4, pady=1)
        badge_pro.pack(side=tk.LEFT, padx=8)

        subtitle_lbl = tk.Label(title_frame,
                                text="|  Franka Panda 7-DOF Digital Twin & Interactive Workcell",
                                font=("Segoe UI", 9),
                                bg=self.colors["sidebar"],
                                fg=self.colors["text_muted"])
        subtitle_lbl.pack(side=tk.LEFT, padx=4)

        # Right Action & Status Controls in Header
        right_hdr = tk.Frame(header, bg=self.colors["sidebar"])
        right_hdr.pack(side=tk.RIGHT)

        # Emergency Stop Button (Safety Halt)
        self.btn_estop = tk.Button(right_hdr,
                                   text="🛑 E-STOP",
                                   font=("Segoe UI", 8, "bold"),
                                   bg=self.colors["red_bg"],
                                   fg="#fca5a5",
                                   activebackground=self.colors["red"],
                                   activeforeground="#ffffff",
                                   relief=tk.FLAT,
                                   padx=10, pady=4,
                                   cursor="hand2",
                                   command=self._on_estop)
        self.btn_estop.pack(side=tk.RIGHT, padx=4)

        # Pop-out / Embedded View Mode Toggle Button
        self.btn_view_mode = tk.Button(right_hdr,
                                       text="🗖 Pop Out 3D",
                                       font=("Segoe UI", 8, "bold"),
                                       bg=self.colors["card"],
                                       fg=self.colors["text"],
                                       relief=tk.FLAT,
                                       padx=10, pady=4,
                                       cursor="hand2",
                                       activebackground=self.colors["card_border"],
                                       command=self._toggle_view_mode)
        self.btn_view_mode.pack(side=tk.RIGHT, padx=4)

        # Dedicated MuJoCo Physics & GPU Pill
        self.engine_pill = tk.Label(right_hdr,
                                    text="⚡ MuJoCo PBR (RTX 2050)",
                                    font=("Segoe UI", 8, "bold"),
                                    bg=self.colors["green_bg"],
                                    fg=self.colors["green"],
                                    padx=10, pady=4)
        self.engine_pill.pack(side=tk.RIGHT, padx=4)

        # Status Pill
        self.status_pill = tk.Label(right_hdr,
                                    text="● SYSTEM READY",
                                    font=("Segoe UI", 8, "bold"),
                                    bg="#1e293b",
                                    fg="#38bdf8",
                                    padx=10, pady=4)
        self.status_pill.pack(side=tk.RIGHT, padx=4)

        # Main Split Body: Left 3D Viewport Pane, Right Studio Tools Pane
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.colors["bg"],
                                           sashrelief=tk.FLAT, sashwidth=6)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 6))

        # LEFT PANE: 3D Viewport Panel
        self.viewport_pane = tk.Frame(self.paned_window, bg=self.colors["card"],
                                      highlightbackground=self.colors["card_border"], highlightthickness=1)
        self.paned_window.add(self.viewport_pane, width=580, minsize=400)

        # RIGHT PANE: Studio Tools Notebook Panel
        self.studio_pane = tk.Frame(self.paned_window, bg=self.colors["bg"])
        self.paned_window.add(self.studio_pane, width=780, minsize=500)

        # Construct Left and Right Panes
        self._build_3d_viewport(self.viewport_pane)

        # Right Pane Tabs
        self.notebook = ttk.Notebook(self.studio_pane)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.tab_game = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.tab_builder = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.tab_hardware = tk.Frame(self.notebook, bg=self.colors["bg"])
        self.tab_telemetry = tk.Frame(self.notebook, bg=self.colors["bg"])

        self.notebook.add(self.tab_game, text=" 🎮 Fulfillment Dispatch ")
        self.notebook.add(self.tab_builder, text=" 🏗️ Workcell Layout Studio ")
        self.notebook.add(self.tab_hardware, text=" 🦾 Hardware Deployment (FCI) ")
        self.notebook.add(self.tab_telemetry, text=" 📊 7-DOF Telemetry ")

        self._build_game_tab()
        self._build_builder_tab()
        self._build_hardware_tab()
        self._build_telemetry_tab()

        # Bottom Status Bar
        self.bottom_bar = tk.Frame(self.root, bg=self.colors["sidebar"], height=28, padx=14, pady=4)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = tk.Label(self.bottom_bar,
                                   text="System initialized. MuJoCo PBR physics active.",
                                   font=("Segoe UI", 9),
                                   bg=self.colors["sidebar"],
                                   fg=self.colors["text_muted"])
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_busy = tk.Label(self.bottom_bar,
                                 text="[IDLE]",
                                 font=("Segoe UI", 9, "bold"),
                                 bg=self.colors["sidebar"],
                                 fg=self.colors["green"])
        self.lbl_busy.pack(side=tk.RIGHT)

    def _on_estop(self):
        """Emergency stop: halts robot immediately and commands safe home pose."""
        self._log_event("EMERGENCY STOP TRIGGERED: Halting motion and restoring safe home posture.")
        self.sim.send_command("RESET_SCENE")
        self.lbl_busy.config(text="[E-STOP RESET]", fg=self.colors["red"])

    def _build_3d_viewport(self, parent):
        """Construct the live embedded 3D simulation viewport with interactive camera controls."""
        # Top Header Bar of 3D Viewport
        vp_head = tk.Frame(parent, bg=self.colors["card"], padx=12, pady=8)
        vp_head.pack(fill=tk.X)

        vp_title = tk.Label(vp_head, text="🎥 3D DIGITAL TWIN VIEWPORT",
                            font=("Segoe UI", 10, "bold"), bg=self.colors["card"], fg=self.colors["accent_light"])
        vp_title.pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(vp_head, text="● 60 FPS (RTX 2050)", font=("Segoe UI", 8, "bold"),
                                bg=self.colors["green_bg"], fg=self.colors["green"], padx=8, pady=2)
        self.lbl_fps.pack(side=tk.LEFT, padx=10)

        # Quick Camera Presets in Viewport Header
        cam_bar = tk.Frame(vp_head, bg=self.colors["card"])
        cam_bar.pack(side=tk.RIGHT)

        tk.Label(cam_bar, text="View: ", font=("Segoe UI", 8), bg=self.colors["card"], fg=self.colors["text_muted"]).pack(side=tk.LEFT, padx=(0, 2))

        self.cam_preset_btns = {}
        for label, preset in [("Isometric", "isometric"), ("Conveyor", "conveyor"),
                              ("Front", "front"), ("Top", "top_down")]:
            btn = tk.Button(cam_bar, text=label, font=("Segoe UI", 8, "bold"),
                            bg=self.colors["sidebar"], fg=self.colors["text_muted"],
                            activebackground=self.colors["accent"], activeforeground="#ffffff",
                            relief=tk.FLAT, padx=7, pady=2, cursor="hand2",
                            command=lambda p=preset: self._set_active_camera(p))
            btn.pack(side=tk.LEFT, padx=2)
            self.cam_preset_btns[preset] = btn

        if "isometric" in self.cam_preset_btns:
            self.cam_preset_btns["isometric"].config(bg=self.colors["accent"], fg="#ffffff")

        # Main 3D Display Frame / Label
        self.viewport_container = tk.Frame(parent, bg="#080a0f", highlightbackground=self.colors["card_border"], highlightthickness=1)
        self.viewport_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.viewport_label = tk.Label(
            self.viewport_container,
            bg="#080a0f",
            text="Initializing 3D Digital Twin Simulation (MuJoCo PBR)...",
            font=("Segoe UI", 10),
            fg=self.colors["text_muted"],
            cursor="fleur"
        )
        self.viewport_label.pack(fill=tk.BOTH, expand=True)

        # Mouse Bindings for Camera Controls on 3D Viewport
        self.viewport_label.bind("<Button-1>", self._on_vp_press)
        self.viewport_label.bind("<B1-Motion>", self._on_vp_orbit)
        self.viewport_label.bind("<ButtonRelease-1>", self._on_vp_release)
        self.viewport_label.bind("<Button-3>", self._on_vp_pan_press)
        self.viewport_label.bind("<B3-Motion>", self._on_vp_pan)
        self.viewport_label.bind("<ButtonRelease-3>", self._on_vp_pan_release)
        self.viewport_label.bind("<MouseWheel>", self._on_vp_zoom)
        self.viewport_label.bind("<Leave>", self._on_vp_leave)
        self.viewport_container.bind("<Leave>", self._on_vp_leave)

        # Bottom HUD Controls Guide & Robot State
        hud_bar = tk.Frame(parent, bg=self.colors["card"], padx=12, pady=6)
        hud_bar.pack(fill=tk.X, side=tk.BOTTOM)

        guide_lbl = tk.Label(hud_bar, text="🖱️ Orbit: Left Drag  |  Pan: Right Drag  |  Zoom: Scroll",
                             font=("Segoe UI", 8), bg=self.colors["card"], fg=self.colors["text_dim"])
        guide_lbl.pack(side=tk.LEFT)

        self.lbl_vp_ee = tk.Label(hud_bar, text="EE: [0.55, 0.00, 0.35] m",
                                  font=("Consolas", 8, "bold"), bg=self.colors["card"], fg=self.colors["accent_light"])
        self.lbl_vp_ee.pack(side=tk.RIGHT, padx=6)

        self.lbl_vp_grip = tk.Label(hud_bar, text="Gripper: OPEN (40mm)",
                                    font=("Consolas", 8, "bold"), bg=self.colors["card"], fg=self.colors["green"])
        self.lbl_vp_grip.pack(side=tk.RIGHT, padx=6)

    def _set_active_camera(self, preset: str):
        """Set camera preset and highlight active button."""
        self.sim.send_command("SET_CAMERA", preset)
        for p, btn in getattr(self, "cam_preset_btns", {}).items():
            if p == preset:
                btn.config(bg=self.colors["accent"], fg="#ffffff")
            else:
                btn.config(bg=self.colors["sidebar"], fg=self.colors["text_muted"])

    def _on_vp_press(self, event):
        self._vp_is_orbiting = True
        self._vp_drag_start_x = event.x
        self._vp_drag_start_y = event.y

    def _on_vp_release(self, event):
        self._vp_is_orbiting = False
        self._vp_drag_start_x = None
        self._vp_drag_start_y = None

    def _on_vp_orbit(self, event):
        if not getattr(self, "_vp_is_orbiting", False):
            return
        if self._vp_drag_start_x is None or self._vp_drag_start_y is None:
            self._vp_drag_start_x = event.x
            self._vp_drag_start_y = event.y
            return
        dx = event.x - self._vp_drag_start_x
        dy = event.y - self._vp_drag_start_y
        self._vp_drag_start_x = event.x
        self._vp_drag_start_y = event.y

        # Discard extreme spikes from pointer warping or uncaptured events
        if abs(dx) > 60 or abs(dy) > 60:
            return

        # Invert horizontal camera movement as requested for natural orbit navigation
        d_az = -dx * 0.45
        d_el = -dy * 0.35
        self.sim.send_command("ORBIT_CAM", (d_az, d_el))

    def _on_vp_pan_press(self, event):
        self._vp_is_panning = True
        self._vp_pan_start_x = event.x
        self._vp_pan_start_y = event.y

    def _on_vp_pan_release(self, event):
        self._vp_is_panning = False
        self._vp_pan_start_x = None
        self._vp_pan_start_y = None

    def _on_vp_pan(self, event):
        if not getattr(self, "_vp_is_panning", False):
            return
        if self._vp_pan_start_x is None or self._vp_pan_start_y is None:
            self._vp_pan_start_x = event.x
            self._vp_pan_start_y = event.y
            return
        dx = event.x - self._vp_pan_start_x
        dy = event.y - self._vp_pan_start_y
        self._vp_pan_start_x = event.x
        self._vp_pan_start_y = event.y

        if abs(dx) > 60 or abs(dy) > 60:
            return

        d_x = -dx * 0.003
        d_y = dy * 0.003
        self.sim.send_command("PAN_CAM", (d_x, d_y))

    def _on_vp_zoom(self, event):
        d_dist = -0.15 if event.delta > 0 else 0.15
        self.sim.send_command("ZOOM_CAM", d_dist)

    def _on_vp_leave(self, event):
        self._vp_is_orbiting = False
        self._vp_is_panning = False
        self._vp_drag_start_x = None
        self._vp_drag_start_y = None
        self._vp_pan_start_x = None
        self._vp_pan_start_y = None

    def _toggle_view_mode(self):
        """Toggle between unified embedded 3D viewport and floating native GLFW window."""
        self.sim.send_command("TOGGLE_VIEW_MODE")

    def _on_view_mode_changed(self, is_embedded: bool):
        """Update UI styling when switching between embedded and pop-out view modes."""
        if is_embedded:
            self.btn_view_mode.config(text="🗖 Pop Out 3D", bg=self.colors["card"], fg=self.colors["text"])
            self.lbl_fps.config(text="● 60 FPS (RTX 2050)", bg=self.colors["green_bg"], fg=self.colors["green"])
            self.viewport_label.config(text="")
        else:
            self.btn_view_mode.config(text="🗕 Embed 3D", bg="#1e293b", fg=self.colors["accent_light"])
            self.lbl_fps.config(text="● POPPED OUT (GLFW)", bg="#1e293b", fg=self.colors["accent_light"])
            self.viewport_label.config(
                image="",
                text="3D SIMULATION IS RUNNING IN POP-OUT WINDOW (GLFW OpenGL)\n\nClick '🗕 Embed 3D' to dock back into unified workspace."
            )

    # -----------------------------------------------------------------------
    # TAB 1: Warehouse Game & Dispatch Catalog
    # -----------------------------------------------------------------------
    def _build_game_tab(self):
        """Construct the Interactive Box Selection & Warehouse Dispatch Game."""
        # Top HUD Dashboard: 4 High-Tech Industrial Metric Cards
        hud_frame = tk.Frame(self.tab_game, bg=self.colors["bg"])
        hud_frame.pack(fill=tk.X, padx=12, pady=(10, 6))

        # 1. Total Score Card
        card_score = tk.Frame(hud_frame, bg=self.colors["card"], padx=14, pady=10,
                              highlightbackground=self.colors["card_border"], highlightthickness=1)
        card_score.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(card_score, text="TOTAL FULFILLMENT SCORE", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_score_val = tk.Label(card_score, text="0", font=("Segoe UI", 18, "bold"),
                                      bg=self.colors["card"], fg=self.colors["amber"])
        self.lbl_score_val.pack(anchor=tk.W)
        self.lbl_combo_badge = tk.Label(card_score, text="Multiplier: 1x (Standard)", font=("Segoe UI", 8),
                                        bg=self.colors["card"], fg=self.colors["accent_light"])
        self.lbl_combo_badge.pack(anchor=tk.W)

        # 2. Completed Orders Card
        card_orders = tk.Frame(hud_frame, bg=self.colors["card"], padx=14, pady=10,
                               highlightbackground=self.colors["card_border"], highlightthickness=1)
        card_orders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(card_orders, text="ORDERS DISPATCHED", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_orders_val = tk.Label(card_orders, text="0 / 6", font=("Segoe UI", 18, "bold"),
                                       bg=self.colors["card"], fg=self.colors["green"])
        self.lbl_orders_val.pack(anchor=tk.W)
        self.order_pbar = ttk.Progressbar(card_orders, orient=tk.HORIZONTAL, length=120, mode="determinate")
        self.order_pbar["maximum"] = 6
        self.order_pbar["value"] = 0
        self.order_pbar.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

        # 3. Motion Quality Card
        card_motion = tk.Frame(hud_frame, bg=self.colors["card"], padx=14, pady=10,
                               highlightbackground=self.colors["card_border"], highlightthickness=1)
        card_motion.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(card_motion, text="MOTION QUALITY RATING", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        tk.Label(card_motion, text="A+ Minimum-Jerk", font=("Segoe UI", 15, "bold"),
                 bg=self.colors["card"], fg=self.colors["accent_light"]).pack(anchor=tk.W)
        tk.Label(card_motion, text="Quintic Spline S-Curve", font=("Segoe UI", 8),
                 bg=self.colors["card"], fg=self.colors["text_dim"]).pack(anchor=tk.W)

        # 4. Workcell Status Card
        card_status = tk.Frame(hud_frame, bg=self.colors["card"], padx=14, pady=10,
                               highlightbackground=self.colors["card_border"], highlightthickness=1)
        card_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(card_status, text="DIGITAL TWIN STATUS", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_cell_status = tk.Label(card_status, text="100% OPERATIONAL", font=("Segoe UI", 15, "bold"),
                                        bg=self.colors["card"], fg=self.colors["green"])
        self.lbl_cell_status.pack(anchor=tk.W)
        tk.Label(card_status, text="Franka FCI Bridge Ready", font=("Segoe UI", 8),
                 bg=self.colors["card"], fg=self.colors["text_dim"]).pack(anchor=tk.W)

        # Center: Order Selection Catalog Cards
        catalog_header = tk.Frame(self.tab_game, bg=self.colors["bg"])
        catalog_header.pack(fill=tk.X, padx=14, pady=(10, 4))

        tk.Label(catalog_header, text="CUSTOMER ORDERS ON INDUSTRIAL SHELF (SELECT BOX TO DISPATCH):",
                 font=("Segoe UI", 10, "bold"), bg=self.colors["bg"], fg=self.colors["text"]).pack(side=tk.LEFT)

        # Quick Actions
        actions_bar = tk.Frame(catalog_header, bg=self.colors["bg"])
        actions_bar.pack(side=tk.RIGHT)

        btn_all = tk.Button(actions_bar, text="⚡ Fulfill All Orders", font=("Segoe UI", 9, "bold"),
                            bg=self.colors["purple"], fg="#ffffff", relief=tk.FLAT, padx=12, pady=4,
                            activebackground=self.colors["purple_hover"], cursor="hand2", command=self._on_fulfill_all)
        btn_all.pack(side=tk.LEFT, padx=3)

        btn_rand = tk.Button(actions_bar, text="🎲 Pick Random", font=("Segoe UI", 9),
                             bg=self.colors["card"], fg=self.colors["text"], relief=tk.FLAT, padx=10, pady=4,
                             activebackground=self.colors["card_border"], cursor="hand2", command=self._on_pick_random)
        btn_rand.pack(side=tk.LEFT, padx=3)

        btn_reset = tk.Button(actions_bar, text="🔄 Restock Shelves", font=("Segoe UI", 9),
                              bg=self.colors["card"], fg=self.colors["text"], relief=tk.FLAT, padx=10, pady=4,
                              activebackground=self.colors["card_border"], cursor="hand2", command=self._on_restock_shelf)
        btn_reset.pack(side=tk.LEFT, padx=3)

        # Grid of Box Cards
        self.cards_container = tk.Frame(self.tab_game, bg=self.colors["bg"])
        self.cards_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Event Activity Log at Bottom of Game Tab
        log_frame = tk.Frame(self.tab_game, bg=self.colors["card"], height=110, padx=12, pady=8,
                             highlightbackground=self.colors["card_border"], highlightthickness=1)
        log_frame.pack(fill=tk.X, padx=12, pady=(4, 8))

        log_head = tk.Frame(log_frame, bg=self.colors["card"])
        log_head.pack(fill=tk.X)

        tk.Label(log_head, text="DISPATCH ACTIVITY & ROBOT EXECUTION LOG:", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_muted"]).pack(side=tk.LEFT)

        btn_clear_log = tk.Button(log_head, text="Clear Log", font=("Segoe UI", 7),
                                  bg=self.colors["sidebar"], fg=self.colors["text_dim"],
                                  relief=tk.FLAT, padx=6, pady=1, cursor="hand2", command=self._clear_log)
        btn_clear_log.pack(side=tk.RIGHT)

        self.txt_log = tk.Text(log_frame, height=4, bg=self.colors["sidebar"], fg=self.colors["text"],
                               font=("Consolas", 8), relief=tk.FLAT, wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._log_event("RoboWarehouse Game Engine initialized.")

    def _clear_log(self):
        """Clear the console log."""
        if hasattr(self, "txt_log"):
            self.txt_log.delete("1.0", tk.END)

    def _render_box_cards(self, boxes: List[Dict]):
        """Render the 6 box order cards dynamically with rich industrial styling."""
        for widget in self.cards_container.winfo_children():
            widget.destroy()
        self.box_widgets.clear()
        self.box_data = boxes

        # 3 columns, 2 rows
        for idx, box in enumerate(boxes):
            row = idx // 3
            col = idx % 3

            card = tk.Frame(self.cards_container, bg=self.colors["card"], padx=14, pady=12,
                            highlightbackground=self.colors["card_border"], highlightthickness=1)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self.cards_container.grid_columnconfigure(col, weight=1)
            self.cards_container.grid_rowconfigure(row, weight=1)

            # Top SKU badge and category pill
            top_bar = tk.Frame(card, bg=self.colors["card"])
            top_bar.pack(fill=tk.X)

            cat_info = [
                ("⚡ ELECTRONICS", "#0284c7", "#e0f2fe"),
                ("🔧 PRECISION TOOLS", "#059669", "#d1fae5"),
                ("📦 PACK & FASTENERS", "#7c3aed", "#ede9fe")
            ]
            cat_tag, cat_color, cat_bg = cat_info[box.get("category", 0) % len(cat_info)]

            lbl_cat = tk.Label(top_bar, text=f" {cat_tag} ", font=("Segoe UI", 7, "bold"),
                               bg=cat_bg, fg=cat_color, padx=3, pady=1)
            lbl_cat.pack(side=tk.LEFT)

            # Status badge
            if box.get("on_conveyor"):
                status_text = "DISPATCHED"
                status_bg = self.colors["green_bg"]
                status_fg = self.colors["green"]
                btn_state = tk.DISABLED
                btn_text = "✓ DELIVERED TO OUTFEED"
                btn_bg = self.colors["sidebar"]
                btn_fg = self.colors["text_dim"]
            elif box.get("on_ground"):
                status_text = "FLOOR RETRIEVAL"
                status_bg = self.colors["red_bg"]
                status_fg = "#fca5a5"
                btn_state = tk.NORMAL
                btn_text = "⚠️ RETRIEVE FROM FLOOR"
                btn_bg = self.colors["amber"]
                btn_fg = "#ffffff"
            else:
                status_text = "ON SHELF"
                status_bg = "#1e293b"
                status_fg = self.colors["accent_light"]
                btn_state = tk.NORMAL
                btn_text = "📦 PICK & DISPATCH"
                btn_bg = self.colors["accent"]
                btn_fg = "#ffffff"

            badge = tk.Label(top_bar, text=f" {status_text} ", font=("Segoe UI", 8, "bold"),
                             bg=status_bg, fg=status_fg, padx=4, pady=1)
            badge.pack(side=tk.RIGHT)

            # SKU and Box Name
            name_frame = tk.Frame(card, bg=self.colors["card"])
            name_frame.pack(fill=tk.X, pady=(8, 2))

            sku_lbl = tk.Label(name_frame, text=box["sku"], font=("Consolas", 10, "bold"),
                               bg=self.colors["card"], fg=self.colors["accent_light"])
            sku_lbl.pack(side=tk.LEFT)

            name_lbl = tk.Label(card, text=box["name"], font=("Segoe UI", 10, "bold"),
                                bg=self.colors["card"], fg=self.colors["text"], anchor=tk.W)
            name_lbl.pack(fill=tk.X, pady=(2, 4))

            # Coordinates / Storage Bay
            pos = box.get("position", [0, 0, 0])
            loc_label = "Floor Retrieval" if box.get("on_ground") else ("Conveyor Belt" if box.get("on_conveyor") else "Shelf Rack")
            loc_text = f"Location: {loc_label} [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
            loc_lbl = tk.Label(card, text=loc_text, font=("Segoe UI", 8),
                               bg=self.colors["card"], fg=self.colors["text_muted"], anchor=tk.W)
            loc_lbl.pack(fill=tk.X, pady=(0, 8))

            # Pick Button
            btn_pick = tk.Button(card, text=btn_text, font=("Segoe UI", 9, "bold"),
                                 bg=btn_bg, fg=btn_fg, relief=tk.FLAT, pady=6,
                                 activebackground=self.colors["accent_hover"],
                                 activeforeground="#ffffff",
                                 cursor="hand2" if btn_state == tk.NORMAL else "arrow",
                                 state=btn_state,
                                 command=lambda b_id=box["body_id"]: self._on_pick_box(b_id))
            btn_pick.pack(fill=tk.X)

            self.box_widgets[box["body_id"]] = {
                "badge": badge,
                "btn": btn_pick,
                "card": card
            }

    def _map_boxes_to_shelves(self):
        """Associate each box with the closest shelf rack and store relative offset."""
        self.box_shelf_map.clear()
        self.box_shelf_offsets.clear()
        shelves = self.current_layout.get("shelves", [])
        if not shelves:
            return
        for box in self.current_layout.get("boxes", []):
            sku = box.get("sku", "")
            bpos = box.get("spawn_position", [0, 0, 0])
            best_idx = 0
            best_dist = float("inf")
            for idx, s in enumerate(shelves):
                spos = s.get("position", [0, 0, 0])
                d = math.hypot(bpos[0] - spos[0], bpos[1] - spos[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            self.box_shelf_map[sku] = best_idx
            spos = shelves[best_idx].get("position", [0, 0, 0])
            self.box_shelf_offsets[sku] = (round(bpos[0] - spos[0], 3), round(bpos[1] - spos[1], 3))

    # -----------------------------------------------------------------------
    # TAB 2: Interactive 2D Workcell Builder (Place Anywhere)
    # -----------------------------------------------------------------------
    def _build_builder_tab(self):
        """Construct the Interactive 2D drag-and-drop factory floor layout editor."""
        left_panel = tk.Frame(self.tab_builder, bg=self.colors["sidebar"], width=330, padx=14, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 6), pady=10)

        # Header Title
        tk.Label(left_panel, text="📐 WORKCELL TOPOLOGY & CAD", font=("Segoe UI", 10, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["accent_light"]).pack(anchor=tk.W, pady=(0, 2))

        tk.Label(left_panel,
                 text="Drag elements on the floorplan to reposition. Dynamic kinematics envelopes verify Franka reach in real time.",
                 font=("Segoe UI", 8), bg=self.colors["sidebar"], fg=self.colors["text_muted"],
                 wraplength=290, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

        # Presets Card
        preset_card = tk.Frame(left_panel, bg=self.colors["card"], padx=8, pady=6,
                               highlightbackground=self.colors["card_border"], highlightthickness=1)
        preset_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(preset_card, text="CELL TOPOLOGY PRESETS:", font=("Segoe UI", 7, "bold"),
                 bg=self.colors["card"], fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 4))

        preset_box = tk.Frame(preset_card, bg=self.colors["card"])
        preset_box.pack(fill=tk.X)

        btn_p1 = tk.Button(preset_box, text="Standard Cell", font=("Segoe UI", 8, "bold"),
                           bg=self.colors["sidebar"], fg=self.colors["text"], relief=tk.FLAT, padx=7, pady=4,
                           activebackground=self.colors["accent"], activeforeground="#ffffff",
                           cursor="hand2", command=lambda: self._apply_preset("standard"))
        btn_p1.pack(side=tk.LEFT, padx=2)

        btn_p2 = tk.Button(preset_box, text="90° L-Cell", font=("Segoe UI", 8, "bold"),
                           bg=self.colors["sidebar"], fg=self.colors["text"], relief=tk.FLAT, padx=7, pady=4,
                           activebackground=self.colors["accent"], activeforeground="#ffffff",
                           cursor="hand2", command=lambda: self._apply_preset("side_conveyor"))
        btn_p2.pack(side=tk.LEFT, padx=2)

        btn_p3 = tk.Button(preset_box, text="Compact Cell", font=("Segoe UI", 8, "bold"),
                           bg=self.colors["sidebar"], fg=self.colors["text"], relief=tk.FLAT, padx=7, pady=4,
                           activebackground=self.colors["accent"], activeforeground="#ffffff",
                           cursor="hand2", command=lambda: self._apply_preset("compact"))
        btn_p3.pack(side=tk.LEFT, padx=2)

        # Coordinate adjustments container
        sliders_box = tk.Frame(left_panel, bg=self.colors["sidebar"])
        sliders_box.pack(fill=tk.X, pady=(0, 6))

        # Robot Arm Base X, Y
        tk.Label(sliders_box, text="🤖 ROBOT ARM BASE (X, Y):", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["accent_light"]).pack(anchor=tk.W, pady=(3, 1))

        arm_pos = self.current_layout.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
        self.slider_arm_x = self._create_slider(sliders_box, "Arm Base X", 0.1, 1.3, arm_pos[0], self._on_slider_change)
        self.slider_arm_y = self._create_slider(sliders_box, "Arm Base Y", -0.7, 0.7, arm_pos[1], self._on_slider_change)

        # Conveyor Belt X, Y
        tk.Label(sliders_box, text="═ OUTFEED CONVEYOR (X, Y):", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["green"]).pack(anchor=tk.W, pady=(5, 1))

        conv_pos = self.current_layout.get("conveyor", {}).get("position", [1.05, 0.0, 0.22])
        self.slider_conv_x = self._create_slider(sliders_box, "Conveyor X", 0.3, 1.6, conv_pos[0], self._on_slider_change)
        self.slider_conv_y = self._create_slider(sliders_box, "Conveyor Y", -0.8, 0.8, conv_pos[1], self._on_slider_change)

        # Shelf Controls
        tk.Label(sliders_box, text="🗄 STORAGE RACKS (X, Y):", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["amber"]).pack(anchor=tk.W, pady=(5, 1))

        shelves = self.current_layout.get("shelves", [])
        s0_pos = shelves[0]["position"] if len(shelves) > 0 else [0.30, -0.30, 0.30]
        s1_pos = shelves[1]["position"] if len(shelves) > 1 else [0.30, 0.30, 0.30]

        self.slider_s0_x = self._create_slider(sliders_box, "Rack 1 X", 0.1, 1.3, s0_pos[0], self._on_slider_change)
        self.slider_s0_y = self._create_slider(sliders_box, "Rack 1 Y", -0.8, 0.8, s0_pos[1], self._on_slider_change)

        self.slider_s1_x = self._create_slider(sliders_box, "Rack 2 X", 0.1, 1.3, s1_pos[0], self._on_slider_change)
        self.slider_s1_y = self._create_slider(sliders_box, "Rack 2 Y", -0.8, 0.8, s1_pos[1], self._on_slider_change)

        # Selected Entity Inspector Card
        inspect_card = tk.Frame(left_panel, bg=self.colors["card"], padx=10, pady=8,
                                highlightbackground=self.colors["card_border"], highlightthickness=1)
        inspect_card.pack(fill=tk.X, pady=(4, 6))

        tk.Label(inspect_card, text="SELECTED ELEMENT INSPECTOR", font=("Segoe UI", 7, "bold"),
                 bg=self.colors["card"], fg=self.colors["accent_light"]).pack(anchor=tk.W)

        self.lbl_inspect_title = tk.Label(inspect_card, text="Franka Panda Cobot Arm",
                                          font=("Segoe UI", 9, "bold"), bg=self.colors["card"], fg=self.colors["text"])
        self.lbl_inspect_title.pack(anchor=tk.W, pady=(2, 0))

        self.lbl_inspect_pos = tk.Label(inspect_card, text="Coordinates: [0.70m, 0.00m]",
                                        font=("Consolas", 8), bg=self.colors["card"], fg=self.colors["text_muted"])
        self.lbl_inspect_pos.pack(anchor=tk.W)

        self.lbl_inspect_reach = tk.Label(inspect_card, text="Working Reach: Center Base [0.00m]",
                                          font=("Segoe UI", 8), bg=self.colors["card"], fg=self.colors["green"])
        self.lbl_inspect_reach.pack(anchor=tk.W, pady=(1, 4))

        # Precision Nudge Buttons
        nudge_row = tk.Frame(inspect_card, bg=self.colors["card"])
        nudge_row.pack(fill=tk.X, pady=2)
        tk.Label(nudge_row, text="Nudge (20mm):", font=("Segoe UI", 7, "bold"), bg=self.colors["card"], fg=self.colors["text_dim"]).pack(side=tk.LEFT, padx=(0, 4))
        for label, dx, dy in [("◀ -X", -0.02, 0), ("▶ +X", 0.02, 0), ("▼ -Y", 0, -0.02), ("▲ +Y", 0, 0.02)]:
            btn = tk.Button(nudge_row, text=label, font=("Consolas", 7, "bold"), bg=self.colors["sidebar"], fg=self.colors["text"],
                            relief=tk.FLAT, padx=5, pady=2, activebackground=self.colors["accent"], cursor="hand2",
                            command=lambda x=dx, y=dy: self._nudge_selected(x, y))
            btn.pack(side=tk.LEFT, padx=1)

        # Action Buttons: Apply & Save
        btn_apply = tk.Button(left_panel, text="🚀 Apply Layout to 3D Simulation", font=("Segoe UI", 9, "bold"),
                              bg=self.colors["accent"], fg="#ffffff", relief=tk.FLAT, pady=7,
                              activebackground=self.colors["accent_hover"], cursor="hand2", command=self._apply_layout_to_sim)
        btn_apply.pack(fill=tk.X, pady=(4, 2))

        btn_save = tk.Button(left_panel, text="💾 Save to layout.json", font=("Segoe UI", 8, "bold"),
                             bg=self.colors["card"], fg=self.colors["text"], relief=tk.FLAT, pady=5,
                             highlightbackground=self.colors["card_border"], highlightthickness=1,
                             activebackground=self.colors["card_hover"], cursor="hand2", command=self._save_layout_to_disk)
        btn_save.pack(fill=tk.X, pady=2)

        # Canvas on the Right
        canvas_frame = tk.Frame(self.tab_builder, bg=self.colors["card"],
                                highlightbackground=self.colors["card_border"], highlightthickness=1)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 10), pady=10)

        # Canvas Header
        c_head = tk.Frame(canvas_frame, bg=self.colors["card"], padx=12, pady=7)
        c_head.pack(fill=tk.X)
        tk.Label(c_head, text="TOP-DOWN WORKCELL BLUEPRINT (1m = 220px)",
                 font=("Segoe UI", 9, "bold"), bg=self.colors["card"], fg=self.colors["text"]).pack(side=tk.LEFT)
        tk.Label(c_head, text="[SCALE 1:4.5 | METRIC CARTESIAN]",
                 font=("Consolas", 8), bg=self.colors["card"], fg=self.colors["text_dim"]).pack(side=tk.LEFT, padx=10)

        self.lbl_reach_status = tk.Label(c_head, text="● 100% KINEMATICALLY REACHABLE",
                                         font=("Segoe UI", 9, "bold"), bg=self.colors["card"], fg=self.colors["green"])
        self.lbl_reach_status.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0e14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas Mouse Bindings for Drag & Drop
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", lambda e: self._draw_canvas())

        self._draw_canvas()
        self._update_inspector()

    def _create_slider(self, parent, label_text, from_val, to_val, init_val, callback):
        """Helper to create labeled slider with live telemetry coordinate readout."""
        f = tk.Frame(parent, bg=self.colors["sidebar"])
        f.pack(fill=tk.X, pady=1)

        hdr = tk.Frame(f, bg=self.colors["sidebar"])
        hdr.pack(fill=tk.X)
        lbl_title = tk.Label(hdr, text=label_text, font=("Segoe UI", 8),
                             bg=self.colors["sidebar"], fg=self.colors["text_muted"])
        lbl_title.pack(side=tk.LEFT)

        lbl_val = tk.Label(hdr, text=f"{init_val:.2f} m", font=("Consolas", 8, "bold"),
                           bg=self.colors["sidebar"], fg=self.colors["accent_light"])
        lbl_val.pack(side=tk.RIGHT)

        def _on_move(v):
            lbl_val.config(text=f"{float(v):.2f} m")
            callback()

        scale = tk.Scale(f, from_=from_val, to=to_val, resolution=0.02, orient=tk.HORIZONTAL,
                         bg=self.colors["sidebar"], fg=self.colors["text"],
                         troughcolor=self.colors["card"], highlightthickness=0,
                         activebackground=self.colors["accent_light"], bd=0,
                         sliderrelief=tk.FLAT, sliderlength=14,
                         showvalue=0, command=_on_move)
        scale.set(init_val)
        scale.pack(fill=tk.X, pady=(1, 2))
        return scale

    def _sync_sliders_from_layout(self):
        """Synchronize UI slider positions with current layout values without triggering loops."""
        self._is_updating_sliders = True
        try:
            arm_pos = self.current_layout.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
            self.slider_arm_x.set(arm_pos[0])
            self.slider_arm_y.set(arm_pos[1])
            conv_pos = self.current_layout.get("conveyor", {}).get("position", [1.05, 0.0, 0.22])
            self.slider_conv_x.set(conv_pos[0])
            self.slider_conv_y.set(conv_pos[1])

            shelves = self.current_layout.get("shelves", [])
            if len(shelves) > 0 and hasattr(self, "slider_s0_x"):
                self.slider_s0_x.set(shelves[0]["position"][0])
                self.slider_s0_y.set(shelves[0]["position"][1])
            if len(shelves) > 1 and hasattr(self, "slider_s1_x"):
                self.slider_s1_x.set(shelves[1]["position"][0])
                self.slider_s1_y.set(shelves[1]["position"][1])
        finally:
            self._is_updating_sliders = False

    def _on_slider_change(self):
        """Update layout when sliders are moved manually."""
        if self._is_updating_sliders:
            return

        self.current_layout["arm"]["base_position"][0] = float(self.slider_arm_x.get())
        self.current_layout["arm"]["base_position"][1] = float(self.slider_arm_y.get())
        self.current_layout["conveyor"]["position"][0] = float(self.slider_conv_x.get())
        self.current_layout["conveyor"]["position"][1] = float(self.slider_conv_y.get())

        shelves = self.current_layout.get("shelves", [])
        if len(shelves) > 0 and hasattr(self, "slider_s0_x"):
            new_x = float(self.slider_s0_x.get())
            new_y = float(self.slider_s0_y.get())
            shelves[0]["position"][0] = new_x
            shelves[0]["position"][1] = new_y
            for box in self.current_layout.get("boxes", []):
                sku = box["sku"]
                if self.box_shelf_map.get(sku) == 0:
                    off_x, off_y = self.box_shelf_offsets.get(sku, (0.09, 0.0))
                    box["spawn_position"][0] = round(new_x + off_x, 3)
                    box["spawn_position"][1] = round(new_y + off_y, 3)

        if len(shelves) > 1 and hasattr(self, "slider_s1_x"):
            new_x = float(self.slider_s1_x.get())
            new_y = float(self.slider_s1_y.get())
            shelves[1]["position"][0] = new_x
            shelves[1]["position"][1] = new_y
            for box in self.current_layout.get("boxes", []):
                sku = box["sku"]
                if self.box_shelf_map.get(sku) == 1:
                    off_x, off_y = self.box_shelf_offsets.get(sku, (0.09, 0.0))
                    box["spawn_position"][0] = round(new_x + off_x, 3)
                    box["spawn_position"][1] = round(new_y + off_y, 3)

        self._update_inspector()
        self._draw_canvas()

    def _update_inspector(self):
        """Update the selected object inspector panel with live metadata."""
        if not hasattr(self, "lbl_inspect_title"):
            return

        arm_base = self.current_layout.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
        sel = self.selected_item or "arm"

        if sel == "arm":
            self.lbl_inspect_title.config(text="Franka Emika Panda 7-DOF Cobot")
            self.lbl_inspect_pos.config(text=f"Base Coordinates: [{arm_base[0]:.2f}m, {arm_base[1]:.2f}m]")
            self.lbl_inspect_reach.config(text="Status: Active Robot Center [0.00m]", fg=self.colors["green"])

        elif sel == "conveyor":
            pos = self.current_layout.get("conveyor", {}).get("position", [1.05, 0.0, 0.22])
            dist = math.hypot(pos[0] - arm_base[0], pos[1] - arm_base[1])
            is_ok = 0.22 <= dist <= 0.855
            self.lbl_inspect_title.config(text="Motorized Outfeed Conveyor")
            self.lbl_inspect_pos.config(text=f"Center Coordinates: [{pos[0]:.2f}m, {pos[1]:.2f}m]")
            status_text = f"Reach Distance: {dist:.2f}m ({'REACHABLE' if is_ok else 'OUT OF REACH'})"
            self.lbl_inspect_reach.config(text=status_text, fg=self.colors["green"] if is_ok else self.colors["red"])

        elif sel.startswith("shelf_"):
            idx = int(sel.split("_")[1])
            shelves = self.current_layout.get("shelves", [])
            if idx < len(shelves):
                s = shelves[idx]
                pos = s.get("position", [0.30, 0.0, 0.30])
                dist = math.hypot(pos[0] - arm_base[0], pos[1] - arm_base[1])
                is_ok = 0.22 <= dist <= 0.855
                self.lbl_inspect_title.config(text=f"{s.get('name', 'Industrial Shelf Rack')}")
                self.lbl_inspect_pos.config(text=f"Rack Coordinates: [{pos[0]:.2f}m, {pos[1]:.2f}m]")
                status_text = f"Reach Distance: {dist:.2f}m ({'REACHABLE' if is_ok else 'OUT OF REACH'})"
                self.lbl_inspect_reach.config(text=status_text, fg=self.colors["green"] if is_ok else self.colors["red"])

        elif sel.startswith("box_"):
            sku = sel[4:]
            box_found = None
            for b in self.current_layout.get("boxes", []):
                if b["sku"] == sku:
                    box_found = b
                    break
            if box_found:
                pos = box_found.get("spawn_position", [0, 0, 0])
                dist = math.hypot(pos[0] - arm_base[0], pos[1] - arm_base[1])
                is_ok = 0.22 <= dist <= 0.855
                self.lbl_inspect_title.config(text=f"Box {sku}: {box_found.get('name', '')}")
                self.lbl_inspect_pos.config(text=f"Shelf Position: [{pos[0]:.2f}m, {pos[1]:.2f}m, {pos[2]:.2f}m]")
                status_text = f"Reach Distance: {dist:.2f}m ({'REACHABLE' if is_ok else 'OUT OF REACH'})"
                self.lbl_inspect_reach.config(text=status_text, fg=self.colors["green"] if is_ok else self.colors["red"])

    def _nudge_selected(self, dx: float, dy: float):
        """Precision nudge selected object by small delta in meters."""
        sel = self.selected_item or "arm"
        if sel == "arm":
            pos = self.current_layout["arm"]["base_position"]
            pos[0] = round(max(0.1, min(1.3, pos[0] + dx)), 3)
            pos[1] = round(max(-0.7, min(0.7, pos[1] + dy)), 3)
        elif sel == "conveyor":
            pos = self.current_layout["conveyor"]["position"]
            pos[0] = round(max(0.3, min(1.6, pos[0] + dx)), 3)
            pos[1] = round(max(-0.8, min(0.8, pos[1] + dy)), 3)
        elif sel.startswith("shelf_"):
            idx = int(sel.split("_")[1])
            shelves = self.current_layout.get("shelves", [])
            if idx < len(shelves):
                spos = shelves[idx]["position"]
                spos[0] = round(max(0.1, min(1.3, spos[0] + dx)), 3)
                spos[1] = round(max(-0.8, min(0.8, spos[1] + dy)), 3)
                for box in self.current_layout.get("boxes", []):
                    sku = box["sku"]
                    if self.box_shelf_map.get(sku) == idx:
                        off_x, off_y = self.box_shelf_offsets.get(sku, (0.09, 0.0))
                        box["spawn_position"][0] = round(spos[0] + off_x, 3)
                        box["spawn_position"][1] = round(spos[1] + off_y, 3)
        elif sel.startswith("box_"):
            sku = sel[4:]
            for box in self.current_layout.get("boxes", []):
                if box["sku"] == sku:
                    box["spawn_position"][0] = round(box["spawn_position"][0] + dx, 3)
                    box["spawn_position"][1] = round(box["spawn_position"][1] + dy, 3)
                    shelf_idx = self.box_shelf_map.get(sku, 0)
                    if shelf_idx < len(self.current_layout.get("shelves", [])):
                        spos = self.current_layout["shelves"][shelf_idx]["position"]
                        self.box_shelf_offsets[sku] = (round(box["spawn_position"][0] - spos[0], 3),
                                                      round(box["spawn_position"][1] - spos[1], 3))
                    break

        self._sync_sliders_from_layout()
        self._update_inspector()
        self._draw_canvas()

    def _draw_canvas(self):
        """Redraw the 2D factory floor with CAD-grade blueprint grid, reach envelopes, and placed assets."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 480

        cx = w // 2 - 30
        cy = h // 2
        self.canvas_origin_x = cx
        self.canvas_origin_y = cy

        # 1. High-Precision CAD Blueprint Grid
        # Minor Grid: every 0.1m (approx 22px)
        step_minor = int(0.1 * self.canvas_scale)
        if step_minor > 10:
            for x in range(cx % step_minor, w, step_minor):
                self.canvas.create_line(x, 0, x, h, fill="#121620", width=1)
            for y in range(cy % step_minor, h, step_minor):
                self.canvas.create_line(0, y, w, y, fill="#121620", width=1)

        # Major Grid: every 0.5m (110px) with metric coordinate readouts
        step_major = int(0.5 * self.canvas_scale)
        for x in range(cx % step_major, w, step_major):
            self.canvas.create_line(x, 0, x, h, fill="#1e2738", width=1)
            m_val = (x - cx) / self.canvas_scale
            if abs(m_val) > 0.05:
                self.canvas.create_text(x, h - 10, text=f"{m_val:+.1f}m", fill="#475569", font=("Consolas", 7))

        for y in range(cy % step_major, h, step_major):
            self.canvas.create_line(0, y, w, y, fill="#1e2738", width=1)
            m_val = (cy - y) / self.canvas_scale
            if abs(m_val) > 0.05:
                self.canvas.create_text(18, y, text=f"{m_val:+.1f}m", fill="#475569", font=("Consolas", 7))

        # Coordinate Axes Origin (World 0, 0)
        self.canvas.create_line(cx - 24, cy, cx + 24, cy, fill="#3b82f6", width=2)
        self.canvas.create_line(cx, cy - 24, cx, cy + 24, fill="#3b82f6", width=2)
        self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#38bdf8", outline="")
        self.canvas.create_text(cx + 12, cy - 12, text="(0.0, 0.0)m", fill="#64748b", font=("Consolas", 8, "bold"))

        # Helper coordinate conversion (Cartesian meters -> Canvas pixels)
        def to_canvas(wx, wy):
            return cx + wx * self.canvas_scale, cy - wy * self.canvas_scale

        # Get positions from layout
        arm_base = self.current_layout.get("arm", {}).get("base_position", [0.70, 0.0, 0.0])
        conv = self.current_layout.get("conveyor", {})
        conv_pos = conv.get("position", [1.05, 0.0, 0.22])
        conv_len = conv.get("length", 1.30)
        conv_wid = conv.get("width", 0.24)
        shelves = self.current_layout.get("shelves", [])
        boxes = self.current_layout.get("boxes", [])

        ax, ay = to_canvas(arm_base[0], arm_base[1])

        # 2. Multi-Tier Franka Kinematic Reach Envelopes
        # Outer Max Reach Circle (855mm max envelope)
        r_max = 0.855 * self.canvas_scale
        self.canvas.create_oval(ax - r_max, ay - r_max, ax + r_max, ay + r_max,
                                outline="#38bdf8", width=2, dash=(6, 4))
        self.canvas.create_text(ax, ay - r_max - 8, text="MAX REACH: 855mm",
                                fill="#38bdf8", font=("Consolas", 7, "bold"))

        # High-Dexterity Sweet Zone (650mm)
        r_sweet = 0.650 * self.canvas_scale
        self.canvas.create_oval(ax - r_sweet, ay - r_sweet, ax + r_sweet, ay + r_sweet,
                                outline="#1e40af", width=1, dash=(3, 3))

        # Inner Singularity Deadzone Circle (220mm)
        r_min = 0.220 * self.canvas_scale
        self.canvas.create_oval(ax - r_min, ay - r_min, ax + r_min, ay + r_min,
                                outline="#ef4444", width=1, dash=(4, 4))
        self.canvas.create_text(ax, ay + r_min + 8, text="DEADZONE: 220mm",
                                fill="#ef4444", font=("Consolas", 7, "bold"))

        # 3. Conveyor Belt
        conv_x, conv_y = to_canvas(conv_pos[0], conv_pos[1])
        c_hw = (conv_wid / 2.0) * self.canvas_scale
        c_hl = (conv_len / 2.0) * self.canvas_scale

        conv_dist = math.hypot(conv_pos[0] - arm_base[0], conv_pos[1] - arm_base[1])
        conv_valid = 0.22 <= conv_dist <= 0.855
        conv_color = "#1a2232" if conv_valid else "#3b1515"

        is_conv_selected = (self.selected_item == "conveyor")
        conv_border = "#f59e0b" if is_conv_selected else ("#10b981" if conv_valid else "#ef4444")
        conv_b_width = 3 if is_conv_selected else 2

        self.canvas.create_rectangle(conv_x - c_hw, conv_y - c_hl, conv_x + c_hw, conv_y + c_hl,
                                     fill=conv_color, outline=conv_border, width=conv_b_width,
                                     tags=("draggable", "conveyor"))

        # Conveyor roller details
        for step_r in range(int(conv_y - c_hl + 15), int(conv_y + c_hl), 24):
            self.canvas.create_line(conv_x - c_hw + 3, step_r, conv_x + c_hw - 3, step_r,
                                    fill="#334155", width=1, tags=("draggable", "conveyor"))

        # Directional feed chevrons
        for arrow_y in [conv_y - c_hl / 2, conv_y, conv_y + c_hl / 2]:
            self.canvas.create_line(conv_x - 6, arrow_y + 4, conv_x, arrow_y - 4, fill="#64748b", width=2,
                                    tags=("draggable", "conveyor"))
            self.canvas.create_line(conv_x + 6, arrow_y + 4, conv_x, arrow_y - 4, fill="#64748b", width=2,
                                    tags=("draggable", "conveyor"))

        self.canvas.create_text(conv_x, conv_y - 12, text="OUTFEED CONVEYOR",
                                fill="#ffffff", font=("Segoe UI", 8, "bold"), justify=tk.CENTER,
                                tags=("draggable", "conveyor"))
        self.canvas.create_text(conv_x, conv_y + 8, text=f"[{conv_pos[0]:.2f}m, {conv_pos[1]:.2f}m]",
                                fill="#94a3b8", font=("Consolas", 7), justify=tk.CENTER,
                                tags=("draggable", "conveyor"))

        # 4. Storage Rack Shelves
        for idx, shelf in enumerate(shelves):
            spos = shelf.get("position", [0.30, 0.0, 0.30])
            sx, sy = to_canvas(spos[0], spos[1])
            sw = 0.34 * self.canvas_scale
            sh = 0.44 * self.canvas_scale

            s_dist = math.hypot(spos[0] - arm_base[0], spos[1] - arm_base[1])
            s_valid = (0.22 <= s_dist <= 0.855)
            s_color = "#141b27" if s_valid else "#3b1515"

            is_shelf_selected = (self.selected_item == f"shelf_{idx}")
            s_border = "#f59e0b" if is_shelf_selected else ("#475569" if s_valid else "#ef4444")
            s_b_width = 3 if is_shelf_selected else 2

            self.canvas.create_rectangle(sx - sw / 2, sy - sh / 2, sx + sw / 2, sy + sh / 2,
                                         fill=s_color, outline=s_border, width=s_b_width,
                                         tags=("draggable", f"shelf_{idx}"))

            # Shelf upright column corner markers
            for ox in [-sw / 2 + 4, sw / 2 - 4]:
                for oy in [-sh / 2 + 4, sh / 2 - 4]:
                    self.canvas.create_rectangle(sx + ox - 3, sy + oy - 3, sx + ox + 3, sy + oy + 3,
                                                 fill="#38bdf8", outline="", tags=("draggable", f"shelf_{idx}"))

            # Shelf cantilever pick lip (amber indicator on pick-facing edge)
            lip_x = sx + sw / 2 - 2 if spos[0] < arm_base[0] else sx - sw / 2 + 2
            self.canvas.create_line(lip_x, sy - sh / 2 + 4, lip_x, sy + sh / 2 - 4,
                                    fill="#f59e0b", width=3, tags=("draggable", f"shelf_{idx}"))

            self.canvas.create_text(sx, sy - sh / 2 + 12, text=shelf.get("name", f"Rack {idx+1}").upper(),
                                    fill="#cbd5e1", font=("Segoe UI", 8, "bold"), tags=("draggable", f"shelf_{idx}"))
            self.canvas.create_text(sx, sy + sh / 2 - 10, text=f"[{spos[0]:.2f}m, {spos[1]:.2f}m]",
                                    fill="#64748b", font=("Consolas", 7), tags=("draggable", f"shelf_{idx}"))

        # 5. Inventory Boxes on Shelves
        for box in boxes:
            sku = box.get("sku", "")
            bpos = box.get("spawn_position", [0, 0, 0])
            bx, by = to_canvas(bpos[0], bpos[1])
            b_r = 9
            cat_colors = ["#38bdf8", "#34d399", "#f472b6"]
            b_color = cat_colors[box.get("category", 0) % 3]

            is_box_selected = (self.selected_item == f"box_{sku}")
            b_border = "#f59e0b" if is_box_selected else "#ffffff"
            b_border_w = 2 if is_box_selected else 1

            self.canvas.create_rectangle(bx - b_r, by - b_r, bx + b_r, by + b_r,
                                         fill=b_color, outline=b_border, width=b_border_w,
                                         tags=("draggable", f"box_{sku}"))
            short_sku = sku.split("-")[-1] if "-" in sku else sku[-3:]
            self.canvas.create_text(bx, by, text=short_sku, fill="#0b0e14", font=("Consolas", 7, "bold"),
                                    tags=("draggable", f"box_{sku}"))

        # 6. Franka Robot Arm Base
        is_arm_selected = (self.selected_item == "arm")
        arm_border = "#f59e0b" if is_arm_selected else "#60a5fa"
        arm_border_w = 3 if is_arm_selected else 2

        # Outer pedestal flange
        self.canvas.create_oval(ax - 24, ay - 24, ax + 24, ay + 24,
                                fill="#1e3a8a", outline=arm_border, width=arm_border_w, tags=("draggable", "arm"))
        # Inner rotor hub
        self.canvas.create_oval(ax - 16, ay - 16, ax + 16, ay + 16,
                                fill="#2563eb", outline="#93c5fd", width=1, tags=("draggable", "arm"))
        # Center pivot
        self.canvas.create_oval(ax - 4, ay - 4, ax + 4, ay + 4,
                                fill="#38bdf8", outline="", tags=("draggable", "arm"))

        self.canvas.create_text(ax, ay + 34, text=f"FRANKA BASE [{arm_base[0]:.2f}m, {arm_base[1]:.2f}m]",
                                fill="#94a3b8", font=("Consolas", 7, "bold"), justify=tk.CENTER, tags=("draggable", "arm"))

        # 7. Sleek Blueprint HUD Legend (bottom right)
        leg_x = w - 16
        leg_y = h - 22
        legend_items = [
            ("═ Outfeed", "#10b981"),
            ("📦 SKUs", "#38bdf8"),
            ("⭕ Singularity: 220mm", "#ef4444"),
            ("⭕ Reach: 855mm", "#38bdf8"),
            ("● Franka Base", "#60a5fa")
        ]
        curr_x = leg_x
        for text, col in legend_items:
            t_id = self.canvas.create_text(curr_x, leg_y, text=text, fill=col, font=("Segoe UI", 7, "bold"), anchor=tk.E)
            bbox = self.canvas.bbox(t_id)
            if bbox:
                curr_x = bbox[0] - 14

        # 8. Kinematics Validation Status
        warnings = LayoutManager.validate_reachability(self.current_layout)
        if len(warnings) == 0:
            self.lbl_reach_status.config(text="● 100% KINEMATICALLY REACHABLE", fg=self.colors["green"])
        else:
            self.lbl_reach_status.config(text=f"▲ {len(warnings)} ELEMENT(S) OUT OF REACH", fg=self.colors["red"])

    def _on_canvas_press(self, event):
        """Identify which object is clicked, prioritizing boxes, arm, conveyor, then shelves."""
        items = self.canvas.find_overlapping(event.x - 6, event.y - 6, event.x + 6, event.y + 6)
        if not items:
            self.drag_item = None
            return

        chosen_tag = None
        # Check all overlapping canvas items from top-most to bottom-most
        for item_id in reversed(items):
            tags = self.canvas.gettags(item_id)
            for t in tags:
                if t.startswith("box_") or t == "arm" or t == "conveyor" or t.startswith("shelf_"):
                    chosen_tag = t
                    break
            if chosen_tag:
                break

        if chosen_tag:
            self.drag_item = chosen_tag
            self.selected_item = chosen_tag
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self._update_inspector()
            self._draw_canvas()
        else:
            self.drag_item = None

    def _on_canvas_drag(self, event):
        """Drag object and update coordinates dynamically with smooth responsive feedback."""
        if not self.drag_item:
            return

        dx = (event.x - self.drag_start_x) / self.canvas_scale
        dy = -(event.y - self.drag_start_y) / self.canvas_scale
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        if self.drag_item == "arm":
            pos = self.current_layout["arm"]["base_position"]
            pos[0] = round(max(0.1, min(1.3, pos[0] + dx)), 3)
            pos[1] = round(max(-0.7, min(0.7, pos[1] + dy)), 3)
            self._sync_sliders_from_layout()

        elif self.drag_item == "conveyor":
            pos = self.current_layout["conveyor"]["position"]
            pos[0] = round(max(0.3, min(1.6, pos[0] + dx)), 3)
            pos[1] = round(max(-0.8, min(0.8, pos[1] + dy)), 3)
            self._sync_sliders_from_layout()

        elif self.drag_item.startswith("shelf_"):
            idx = int(self.drag_item.split("_")[1])
            shelves = self.current_layout.get("shelves", [])
            if idx < len(shelves):
                spos = shelves[idx]["position"]
                spos[0] = round(max(0.1, min(1.3, spos[0] + dx)), 3)
                spos[1] = round(max(-0.8, min(0.8, spos[1] + dy)), 3)
                # Synchronously move all boxes mapped to this shelf
                for box in self.current_layout.get("boxes", []):
                    sku = box["sku"]
                    if self.box_shelf_map.get(sku) == idx:
                        off_x, off_y = self.box_shelf_offsets.get(sku, (0.09, 0.0))
                        box["spawn_position"][0] = round(spos[0] + off_x, 3)
                        box["spawn_position"][1] = round(spos[1] + off_y, 3)
                self._sync_sliders_from_layout()

        elif self.drag_item.startswith("box_"):
            sku = self.drag_item[4:]
            for box in self.current_layout.get("boxes", []):
                if box["sku"] == sku:
                    box["spawn_position"][0] = round(box["spawn_position"][0] + dx, 3)
                    box["spawn_position"][1] = round(box["spawn_position"][1] + dy, 3)
                    # Update offset relative to its parent shelf
                    shelf_idx = self.box_shelf_map.get(sku, 0)
                    if shelf_idx < len(self.current_layout.get("shelves", [])):
                        spos = self.current_layout["shelves"][shelf_idx]["position"]
                        self.box_shelf_offsets[sku] = (round(box["spawn_position"][0] - spos[0], 3),
                                                      round(box["spawn_position"][1] - spos[1], 3))
                    break

        self._update_inspector()
        self._draw_canvas()

    def _on_canvas_release(self, event):
        """Finalize drag movement and refresh reachability status."""
        self.drag_item = None
        self._update_inspector()
        self._draw_canvas()

    def _apply_preset(self, preset_name: str):
        """Load and apply workcell layout preset."""
        import copy
        preset = copy.deepcopy(LayoutManager.get_preset(preset_name))
        self.current_layout = preset
        self._map_boxes_to_shelves()
        self._sync_sliders_from_layout()
        self.selected_item = "arm"
        self._update_inspector()
        self._draw_canvas()
        self._log_event(f"Loaded preset: '{preset_name}'. Click 'Apply to 3D Simulation' to see changes in 3D!")

    def _save_layout_to_disk(self):
        """Save layout to layout.json file."""
        LayoutManager.save_layout(self.current_layout, "layout.json")
        messagebox.showinfo("Saved", "Factory workcell layout saved to layout.json successfully!")
        self._log_event("Workcell layout saved to layout.json.")

    def _apply_layout_to_sim(self):
        """Send new layout to the running 3D simulation and save to layout.json."""
        import copy
        LayoutManager.save_layout(self.current_layout, "layout.json")
        self.sim.send_command("RELOAD_LAYOUT", copy.deepcopy(self.current_layout))
        self._log_event("🚀 Applying custom layout to 3D simulation...")

    # -----------------------------------------------------------------------
    # TAB 3: Real Robot Deployment
    # -----------------------------------------------------------------------
    def _build_hardware_tab(self):
        """Construct the Real Franka Panda Physical Robot Deployment panel."""
        top_card = tk.Frame(self.tab_hardware, bg=self.colors["card"], padx=18, pady=16,
                            highlightbackground=self.colors["card_border"], highlightthickness=1)
        top_card.pack(fill=tk.X, padx=14, pady=10)

        tk.Label(top_card, text="FRANKA EMIKA PANDA - PHYSICAL HARDWARE BRIDGE",
                 font=("Segoe UI", 12, "bold"), bg=self.colors["card"], fg=self.colors["accent_light"]).pack(anchor=tk.W)

        tk.Label(top_card,
                 text="Export or stream minimum-jerk trajectories generated by the 3D digital twin directly to a physical Franka Panda arm via Franka Control Interface (FCI) or frankapy at 1000 Hz.",
                 font=("Segoe UI", 9), bg=self.colors["card"], fg=self.colors["text_muted"],
                 wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 12))

        # Connection settings
        conn_frame = tk.Frame(top_card, bg=self.colors["card"])
        conn_frame.pack(fill=tk.X, pady=6)

        tk.Label(conn_frame, text="Robot Controller IP:", font=("Segoe UI", 9, "bold"),
                 bg=self.colors["card"], fg=self.colors["text"]).pack(side=tk.LEFT)

        self.entry_ip = tk.Entry(conn_frame, font=("Consolas", 10), bg=self.colors["sidebar"],
                                 fg=self.colors["text"], insertbackground="#ffffff", width=16)
        self.entry_ip.insert(0, "172.16.0.2")
        self.entry_ip.pack(side=tk.LEFT, padx=10)

        self.lbl_hw_status = tk.Label(conn_frame, text="● STANDBY (Ready for Deployment)",
                                      font=("Segoe UI", 9, "bold"), bg=self.colors["card"], fg=self.colors["amber"])
        self.lbl_hw_status.pack(side=tk.LEFT, padx=10)

        # Action Buttons
        btn_box = tk.Frame(top_card, bg=self.colors["card"])
        btn_box.pack(fill=tk.X, pady=(12, 4))

        btn_exp_json = tk.Button(btn_box, text="💾 Export Trajectory (.JSON)", font=("Segoe UI", 9, "bold"),
                                 bg=self.colors["sidebar"], fg=self.colors["text"], relief=tk.FLAT, padx=12, pady=6,
                                 cursor="hand2", activebackground=self.colors["card_border"],
                                 command=self._on_export_trajectory_json)
        btn_exp_json.pack(side=tk.LEFT, padx=4)

        btn_exp_csv = tk.Button(btn_box, text="📊 Export Trajectory (.CSV)", font=("Segoe UI", 9, "bold"),
                                bg=self.colors["sidebar"], fg=self.colors["text"], relief=tk.FLAT, padx=12, pady=6,
                                cursor="hand2", activebackground=self.colors["card_border"],
                                command=self._on_export_trajectory_csv)
        btn_exp_csv.pack(side=tk.LEFT, padx=4)

        btn_deploy_real = tk.Button(btn_box, text="🚀 Deploy to Physical Arm (FCI)", font=("Segoe UI", 10, "bold"),
                                    bg=self.colors["accent"], fg="#ffffff", relief=tk.FLAT, padx=16, pady=6,
                                    cursor="hand2", activebackground=self.colors["accent_hover"],
                                    command=self._on_deploy_physical_arm)
        btn_deploy_real.pack(side=tk.LEFT, padx=6)

        # Information / Guidance card
        info_card = tk.Frame(self.tab_hardware, bg=self.colors["card"], padx=18, pady=16,
                             highlightbackground=self.colors["card_border"], highlightthickness=1)
        info_card.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        tk.Label(info_card, text="HARDWARE DEPLOYMENT PROTOCOLS & SAFETY CHECKLIST:",
                 font=("Segoe UI", 10, "bold"), bg=self.colors["card"], fg=self.colors["text"]).pack(anchor=tk.W)

        guide_text = (
            "1. Physical Connection: Connect an Ethernet cable directly from your host network adapter to the "
            "Franka Control Box (default subnet 172.16.0.X).\n\n"
            "2. Franka Desk Activation: Open https://172.16.0.2 in browser, unlock joint brakes, "
            "and switch the FCI toggle to ACTIVE.\n\n"
            "3. Safety First: Keep the hardware emergency E-STOP button in hand. The exported trajectories use "
            "minimum-jerk quintic interpolation bounded to cobot safety speeds with zero velocity and acceleration at start/finish.\n\n"
            "4. Digital Twin Execution: When deployed, the 3D MuJoCo PBR simulation visualizes the "
            "arm's synchronized motion in real time while the physical arm carries out the pick-and-place operation."
        )
        tk.Label(info_card, text=guide_text, font=("Segoe UI", 9),
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=8)

    def _on_export_trajectory_json(self):
        """Export joint waypoints to JSON."""
        target_path = "exported_trajectory.json"
        if self.sim.env and hasattr(self.sim.env, "hardware"):
            self.sim.env.hardware.export_trajectory_json(target_path)
            messagebox.showinfo("Export Successful", f"Trajectory exported to: {os.path.abspath(target_path)}")
            self._log_event(f"Hardware trajectory exported to {target_path}")
        else:
            messagebox.showwarning("Notice", "Run a pick action first to record trajectory waypoints!")

    def _on_export_trajectory_csv(self):
        """Export joint waypoints to CSV."""
        target_path = "exported_trajectory.csv"
        if self.sim.env and hasattr(self.sim.env, "hardware"):
            self.sim.env.hardware.export_trajectory_csv(target_path)
            messagebox.showinfo("Export Successful", f"Trajectory CSV exported to: {os.path.abspath(target_path)}")
            self._log_event(f"Hardware trajectory CSV exported to {target_path}")
        else:
            messagebox.showwarning("Notice", "Run a pick action first to record trajectory waypoints!")

    def _on_deploy_physical_arm(self):
        """Deploy to physical robot arm."""
        ip = self.entry_ip.get().strip() or "172.16.0.2"
        confirm = messagebox.askyesno(
            "Confirm Hardware Deployment",
            f"Are you connected to Franka Control Interface at {ip}?\n\n"
            "Ensure the physical robot workspace is clear and the E-Stop is ready. Proceed?"
        )
        if confirm:
            self._log_event(f"Connecting to physical Franka Panda arm at {ip} via FCI...")
            def run_hw():
                os.system(f"python deploy_real_franka.py --robot-ip {ip}")
            threading.Thread(target=run_hw, daemon=True).start()
            messagebox.showinfo("Hardware Deployment Initiated",
                                f"Hardware driver launched targeting Franka Panda at {ip}.\n"
                                "Check terminal logs for real-time FCI communication status.")

    # -----------------------------------------------------------------------
    # TAB 4: Robot Telemetry
    # -----------------------------------------------------------------------
    def _build_telemetry_tab(self):
        """Construct the live robot telemetry monitor."""
        main_box = tk.Frame(self.tab_telemetry, bg=self.colors["bg"])
        main_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Left Column: Joint Angles Table
        col_joints = tk.Frame(main_box, bg=self.colors["card"], padx=18, pady=16,
                              highlightbackground=self.colors["card_border"], highlightthickness=1)
        col_joints.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(col_joints, text="7-DOF COBOT JOINT TELEMETRY", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["card"], fg=self.colors["accent_light"]).pack(anchor=tk.W, pady=(0, 10))

        self.joint_labels = []
        self.joint_bars = []

        joint_info = [
            ("q0 Base Yaw", "±166°"),
            ("q1 Shoulder", "-101° to +101°"),
            ("q2 Elbow Yaw", "±166°"),
            ("q3 Elbow Pitch", "-176° to -4°"),
            ("q4 Wrist Yaw", "±166°"),
            ("q5 Wrist Pitch", "-1° to +215°"),
            ("q6 Flange", "±166°")
        ]

        for i, (name, j_range) in enumerate(joint_info):
            row = tk.Frame(col_joints, bg=self.colors["card"])
            row.pack(fill=tk.X, pady=4)

            lbl_name = tk.Label(row, text=f"{name}:", font=("Segoe UI", 9, "bold"),
                                bg=self.colors["card"], fg=self.colors["text"], width=18, anchor=tk.W)
            lbl_name.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=180, mode="determinate")
            pbar["maximum"] = 360
            pbar["value"] = 180
            pbar.pack(side=tk.LEFT, padx=8)

            lbl_val = tk.Label(row, text="0.00 rad (0.0°)", font=("Consolas", 9),
                               bg=self.colors["card"], fg=self.colors["text_muted"], width=18, anchor=tk.W)
            lbl_val.pack(side=tk.LEFT)

            lbl_range = tk.Label(row, text=f"[{j_range}]", font=("Segoe UI", 7),
                                 bg=self.colors["card"], fg=self.colors["text_dim"])
            lbl_range.pack(side=tk.LEFT, padx=4)

            self.joint_labels.append(lbl_val)
            self.joint_bars.append(pbar)

        # Right Column: End-Effector & Gripper Diagnostics
        col_diag = tk.Frame(main_box, bg=self.colors["card"], padx=18, pady=16,
                            highlightbackground=self.colors["card_border"], highlightthickness=1)
        col_diag.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(col_diag, text="CARTESIAN END-EFFECTOR & GRIPPER", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["card"], fg=self.colors["accent_light"]).pack(anchor=tk.W, pady=(0, 10))

        # Position Card
        pos_box = tk.Frame(col_diag, bg=self.colors["sidebar"], padx=14, pady=10,
                           highlightbackground=self.colors["card_border"], highlightthickness=1)
        pos_box.pack(fill=tk.X, pady=(0, 8))

        tk.Label(pos_box, text="TCP CARTESIAN POSITION (X, Y, Z):", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_ee_pos = tk.Label(pos_box, text="[0.550, 0.000, 0.350] m",
                                   font=("Consolas", 13, "bold"), bg=self.colors["sidebar"], fg=self.colors["accent_light"])
        self.lbl_ee_pos.pack(anchor=tk.W, pady=(2, 0))

        # Gripper Card
        grip_box = tk.Frame(col_diag, bg=self.colors["sidebar"], padx=14, pady=10,
                            highlightbackground=self.colors["card_border"], highlightthickness=1)
        grip_box.pack(fill=tk.X, pady=8)

        tk.Label(grip_box, text="PARALLEL JAW GRIPPER STATUS:", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_gripper = tk.Label(grip_box, text="Gripper Status: OPEN (40mm width)",
                                    font=("Consolas", 11, "bold"), bg=self.colors["sidebar"], fg=self.colors["green"])
        self.lbl_gripper.pack(anchor=tk.W, pady=(2, 0))

        # Safety & Collision Card
        safety_box = tk.Frame(col_diag, bg=self.colors["sidebar"], padx=14, pady=10,
                              highlightbackground=self.colors["card_border"], highlightthickness=1)
        safety_box.pack(fill=tk.X, pady=8)

        tk.Label(safety_box, text="SAFETY & COLLISION MONITOR:", font=("Segoe UI", 8, "bold"),
                 bg=self.colors["sidebar"], fg=self.colors["text_muted"]).pack(anchor=tk.W)
        self.lbl_safety = tk.Label(safety_box, text="● NOMINAL - NO ACTIVE COLLISION",
                                   font=("Segoe UI", 10, "bold"), bg=self.colors["sidebar"], fg=self.colors["green"])
        self.lbl_safety.pack(anchor=tk.W, pady=(2, 0))

    # -----------------------------------------------------------------------
    # Game Logic & Event Handling
    # -----------------------------------------------------------------------
    def _on_pick_box(self, box_id: int):
        """User clicked 'Pick This Box' button."""
        self._log_event(f"Dispatching cobot to retrieve Box {box_id}...")
        self.sim.send_command("PICK_BOX", box_id)

    def _on_fulfill_all(self):
        """User clicked 'Fulfill All Orders'."""
        self._log_event("Starting automated fulfillment sequence for all orders...")
        self.sim.send_command("DISPATCH_ALL")

    def _on_pick_random(self):
        """User clicked 'Pick Random'."""
        available = [b for b in self.box_data if b.get("on_shelf")]
        if not available:
            messagebox.showinfo("Shelf Empty", "All boxes have been dispatched! Click 'Restock Shelf' to play again.")
            return
        chosen = np.random.choice(available)
        self._on_pick_box(chosen["body_id"])

    def _on_restock_shelf(self):
        """Restock all boxes onto industrial shelf."""
        self._log_event("Restocking industrial warehouse shelves...")
        self.sim.send_command("RESET_SCENE")

    def _log_event(self, text: str):
        """Append timestamped message to event log."""
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{timestamp}] {text}\n")
        self.txt_log.see(tk.END)

    def _poll_engine(self):
        """Poll events from background simulation engine."""
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return

        latest_frame = None
        try:
            while True:
                event, data = self.ui_queue.get_nowait()
                if event == "FRAME":
                    latest_frame = data
                elif event == "VIEW_MODE_CHANGED":
                    self._on_view_mode_changed(data)
                else:
                    self._handle_sim_event(event, data)
        except queue.Empty:
            pass

        if latest_frame is not None and getattr(self, "viewport_label", None) is not None:
            try:
                if self.viewport_label.winfo_exists():
                    self.sim_photo = tk.PhotoImage(data=latest_frame, format="PPM")
                    self.viewport_label.config(image=self.sim_photo, text="")
                    self.viewport_label.image = self.sim_photo

                    # Rolling FPS calculation and HUD update
                    self._fps_count += 1
                    now = time.time()
                    elapsed = now - self._fps_last_time
                    if elapsed >= 0.5:
                        self._current_fps = self._fps_count / elapsed
                        self._fps_count = 0
                        self._fps_last_time = now
                        if hasattr(self, "lbl_fps") and self.lbl_fps.winfo_exists():
                            tag = "RTX 2050" if ("RTX" in self.gpu_name or "NVIDIA" in self.gpu_name) else "GPU"
                            self.lbl_fps.config(text=f"● {int(round(self._current_fps))} FPS ({tag})", bg="#064e3b", fg="#34d399")
            except Exception:
                pass

        try:
            if self.root.winfo_exists():
                self._poll_timer_id = self.root.after(16, self._poll_engine)
        except Exception:
            pass

    def _handle_sim_event(self, event: str, data):
        """Process messages from simulation thread."""
        if event == "STATUS":
            self.lbl_status.config(text=str(data))

        elif event == "GPU_DETECTED":
            self.gpu_name = str(data)
            self._log_event(f"Hardware GPU Acceleration: {self.gpu_name}")
            tag = "NVIDIA RTX 2050" if ("RTX" in self.gpu_name or "NVIDIA" in self.gpu_name) else "GPU"
            if hasattr(self, "status_pill") and self.status_pill.winfo_exists():
                self.status_pill.config(text=f"● {tag} ACTIVE", bg="#064e3b", fg="#34d399")
            if hasattr(self, "engine_pill") and self.engine_pill.winfo_exists():
                self.engine_pill.config(text=f"⚡ MuJoCo PBR ({tag})", bg="#064e3b", fg="#34d399")

        elif event == "SIM_READY":
            self.status_pill.config(text="● 3D SIMULATION ACTIVE", bg="#064e3b", fg="#34d399")
            self.lbl_status.config(text="Simulation ready")
            if "boxes" in data:
                self._render_box_cards(data["boxes"])
            self._draw_canvas()

        elif event == "ACTION_START":
            self.lbl_busy.config(text="[BUSY: MOVING]", fg=self.colors["amber"])
            self.lbl_status.config(text=str(data))
            self._log_event(str(data))

        elif event == "ACTION_COMPLETE":
            self.lbl_busy.config(text="[IDLE]", fg=self.colors["green"])
            success = data.get("success", False)
            msg = data.get("message", "")
            self._log_event(f"Action result: {msg}")

            if success:
                self.score += 150 * self.combo
                self.combo = min(5, self.combo + 1)
                self.orders_fulfilled += 1
                self.lbl_score_val.config(text=str(self.score))
                self.lbl_orders_val.config(text=f"{self.orders_fulfilled} / 6")

            if "snapshot" in data and "boxes" in data["snapshot"]:
                self._render_box_cards(data["snapshot"]["boxes"])

        elif event == "ALL_COMPLETE":
            self.lbl_busy.config(text="[IDLE]", fg=self.colors["green"])
            stats = data.get("stats", {})
            placed = stats.get("successful_placements", 6)
            self.score += placed * 200
            self.orders_fulfilled = 6
            self.lbl_score_val.config(text=str(self.score))
            self.lbl_orders_val.config(text="6 / 6 [ALL DELIVERED]")
            self._log_event(f"Automated fulfillment complete: {placed} boxes placed successfully!")
            if "snapshot" in data and "boxes" in data["snapshot"]:
                self._render_box_cards(data["snapshot"]["boxes"])

        elif event == "RESET_COMPLETE":
            self.lbl_busy.config(text="[IDLE]", fg=self.colors["green"])
            self.orders_fulfilled = 0
            self.combo = 1
            self.lbl_orders_val.config(text="0 / 6")
            self._log_event("Shelf restocked with full inventory.")
            if "boxes" in data:
                self._render_box_cards(data["boxes"])

        elif event == "LAYOUT_APPLIED":
            self.lbl_busy.config(text="[IDLE]", fg=self.colors["green"])
            self._log_event("New workcell layout successfully applied to 3D world!")
            if "boxes" in data:
                self._render_box_cards(data["boxes"])
            self._draw_canvas()

        elif event == "TELEMETRY":
            joints = data.get("joints", [])
            for i, val in enumerate(joints):
                if i < len(self.joint_labels):
                    deg = math.degrees(val)
                    self.joint_labels[i].config(text=f"{val:.2f} rad ({deg:.1f}°)")
                    # map [-pi, pi] to [0, 360]
                    norm_val = ((deg + 180.0) % 360.0)
                    self.joint_bars[i]["value"] = norm_val

            ee = data.get("ee_pos", [0, 0, 0])
            self.lbl_ee_pos.config(text=f"Position [X, Y, Z]: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m")
            if hasattr(self, "lbl_vp_ee"):
                self.lbl_vp_ee.config(text=f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}] m")

            is_open = data.get("is_open", True)
            if is_open:
                self.lbl_gripper.config(text="Gripper Status: OPEN (40mm width)", fg=self.colors["green"])
                if hasattr(self, "lbl_vp_grip"):
                    self.lbl_vp_grip.config(text="Gripper: OPEN", fg=self.colors["green"])
            else:
                self.lbl_gripper.config(text="Gripper Status: GRASPING (50N force)", fg=self.colors["amber"])
                if hasattr(self, "lbl_vp_grip"):
                    self.lbl_vp_grip.config(text="Gripper: GRASPING", fg=self.colors["amber"])

        elif event == "ERROR":
            self.lbl_busy.config(text="[ERROR]", fg=self.colors["red"])
            self._log_event(f"ERROR: {data}")

    def _on_close(self):
        """Clean shutdown of simulation and UI."""
        try:
            if hasattr(self, "_poll_timer_id") and self._poll_timer_id and self.root.winfo_exists():
                self.root.after_cancel(self._poll_timer_id)
        except Exception:
            pass
        self.sim.stop()
        try:
            self.root.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------

def main():
    """Launch the RoboWarehouse Studio desktop application."""
    root = tk.Tk()
    app = RoboWarehouseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

