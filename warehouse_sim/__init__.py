"""
RoboWarehouse Simulation & Digital Twin.

High-fidelity MuJoCo Physically-Based Rendering (PBR) simulation of an industrial
warehouse fulfillment workcell with the Franka Emika Panda 7-DOF arm, interactive
cantilever shelving, conveyor dispatch, and direct physical robot trajectory bridge.
"""

__version__ = "2.0.0"

# Main configuration constants
from .config import (
    BOX_SIZE,
    BIN_POSITIONS,
    SHELF_POSITIONS,
    GRAVITY,
    TIMESTEP,
    CONVEYOR_X,
    CONVEYOR_HEIGHT,
    CONVEYOR_SPEED,
    CONVEYOR_DROP_POSITION,
    get_box_spawn_positions,
    get_bin_for_category,
    get_box_color
)

# Core MuJoCo simulation & Desktop Studio components
from .mujoco_env import MuJoCoWarehouseEnv
from .layout_manager import LayoutManager
from .hardware_bridge import HardwareBridge
from .app import RoboWarehouseApp

__all__ = [
    'MuJoCoWarehouseEnv',
    'RoboWarehouseApp',
    'LayoutManager',
    'HardwareBridge',
    'get_box_spawn_positions',
    'get_bin_for_category',
    'get_box_color',
    'BOX_SIZE',
    'BIN_POSITIONS',
    'SHELF_POSITIONS',
    'CONVEYOR_X',
    'CONVEYOR_HEIGHT',
    'CONVEYOR_SPEED',
    'CONVEYOR_DROP_POSITION',
    'GRAVITY',
    'TIMESTEP',
]