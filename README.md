# RoboWarehouse Pro - Franka Panda Digital Twin & Studio

An industrial-grade robotics workstation and digital twin simulation of an autonomous warehouse fulfillment workcell using Google DeepMind's **MuJoCo** physics and Physically-Based Rendering (PBR) engine, featuring the 7-DOF Franka Emika Panda robotic arm.

---

## 🎯 System Capabilities

- **Unified Robotics Workstation**: Complete desktop application built with Python Tkinter and MuJoCo PBR rendering, eliminating external GUI framework dependencies.
- **Dedicated GPU Acceleration**: Configured for high-performance GPU execution (NVIDIA GeForce RTX) providing smooth 60 FPS viewport rendering.
- **Physical Contact & Gravity Mechanics**: Full rigid-body contact physics, real cantilever shelving, zero box clipping, and ground recovery mechanics for dropped inventory.
- **Smooth Minimum-Jerk Cobot Kinematics**: Quintic polynomial S-curve trajectory planning with collision avoidance and bounded joint velocities.
- **Motorized Outfeed Conveyor**: Real-time conveyor transport delivering fulfilled boxes toward dispatch packaging.
- **Interactive 2D Workcell Blueprint CAD Studio**: Drag-and-drop floorplan layout editor with real-time Franka kinematic reach verification (max reach 855mm, inner deadzone 220mm).
- **Physical Robot Deployment (FCI)**: Direct export and deployment of joint trajectories to physical Franka Emika Panda arms via FCI / `frankapy`.
- **7-DOF Real-Time Telemetry**: Real-time joint angle tracking against physical limits, end-effector coordinates, gripper telemetry, and contact force monitoring.

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 (or Linux)
- Dedicated NVIDIA GPU (recommended for 60 FPS PBR rendering)

### Install Dependencies

```bash
pip install -r requirements.txt
```

*Requirements:*
- `mujoco>=3.0.0`
- `numpy>=1.24`

---

## 🚀 Quick Start

### 1. Launch RoboWarehouse Pro Studio (Recommended)

Double-click **`start_app.bat`** in Windows Explorer, or run:

```bash
python app.py
```

This launches the unified desktop studio:
- **Left Panel (3D Digital Twin Viewport)**: Embedded live 60 FPS MuJoCo PBR simulation with interactive camera controls (Left-Click drag to orbit, Right-Click drag to pan, scroll to zoom) and camera presets (`Isometric`, `Conveyor`, `Front`, `Top`).
- **Tab 1: Fulfillment Dispatch**: Pick customer order boxes, dispatch to conveyor, track orders, combos, and motion quality metrics.
- **Tab 2: Workcell CAD & Topology Studio**: Reposition robot, conveyor, and storage racks with 2D drag-and-drop or sliders, and live kinematics envelope verification.
- **Tab 3: Hardware Deployment (FCI)**: Pre-flight safety checklist, trajectory export (`.JSON` and `.CSV`), and physical robot deployment.
- **Tab 4: 7-DOF Telemetry**: Joint angle limit bars, Cartesian end-effector coordinates, and live collision status.

---

### 2. MuJoCo Interactive Terminal CLI

Run the simulation directly from the terminal:

```bash
# Interactive terminal inventory picker
python -m warehouse_sim.mujoco_main

# Pick a specific box directly (1-6) or by SKU
python -m warehouse_sim.mujoco_main --box 1
python -m warehouse_sim.mujoco_main --box ELEC-101

# Automated full fulfillment episode
python -m warehouse_sim.mujoco_main --all

# Export trajectory for real Franka Panda arm
python -m warehouse_sim.mujoco_main --box 1 --export-trajectory
```

---

## 📁 Project Architecture

```
EDI_improved/
├── app.py                      # Desktop Studio root launcher
├── start_app.bat               # Windows GPU-accelerated 1-click launcher
├── requirements.txt            # Python dependencies (MuJoCo, NumPy)
├── layout.json                 # Workcell layout specification (arm, shelves, conveyor, boxes)
├── deploy_real_franka.py       # Physical Franka hardware deployment runner (FCI / frankapy)
├── exported_trajectory.json    # Exported joint trajectory data (JSON)
├── exported_trajectory.csv     # Exported joint trajectory data (CSV)
├── tests/
│   └── test_warehouse.py       # Unit and integration test suite (11 tests)
└── warehouse_sim/
    ├── __init__.py             # Package exports
    ├── app.py                  # RoboWarehouse Studio GUI & embedded 3D viewport
    ├── mujoco_env.py           # MuJoCo PBR simulation engine & kinematics
    ├── mujoco_main.py          # Standalone CLI simulation runner
    ├── layout_manager.py       # Workcell layout validation & JSON persistence
    ├── hardware_bridge.py      # Trajectory recorder & physical hardware bridge
    └── config.py               # Geometric dimensions, physics constants & catalog
```

---

## 🦾 Physical Hardware Deployment

Trajectories recorded during simulation can be directly executed on a physical Franka Emika Panda arm via the **Franka Control Interface (FCI)**.

### 1. Export Trajectories
From the Studio (Tab 3) or CLI:
```bash
python -m warehouse_sim.mujoco_main --box 1 --export-trajectory
```

### 2. Execute on Real Robot
Connect your workstation to the Franka Control Box via Ethernet and run:
```bash
# Dry run verification
python deploy_real_franka.py

# Live deployment to physical arm
python deploy_real_franka.py --robot-ip 172.16.0.2
```

---

## 🧪 Running Automated Tests

Run the complete 11-test suite covering kinematics, IK reachability, box dispatch, layout persistence, and UI synchronization:

```bash
python -m unittest tests/test_warehouse.py
```

---

## 📄 License
This project is developed for robotics research and educational simulation.
