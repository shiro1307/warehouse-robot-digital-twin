"""
Configuration constants for the warehouse simulation.

This module contains all configurable parameters for the warehouse environment,
including physical dimensions, positions, and simulation settings.

Coordinate Frame Convention:
- X: forward/backward (shelf depth direction)
- Y: left/right (shelf width direction)
- Z: up/down (vertical)
- Origin (0, 0, 0) is at ground level, centered between shelves
"""

import numpy as np

# ============================================================================
# Simulation Physics
# ============================================================================
GRAVITY = (0, 0, -9.81)  # m/s^2
TIMESTEP = 1.0 / 240.0  # seconds (240 Hz)
SUBSTEPS_PER_STEP = 10  # physics steps per environment step

# ============================================================================
# Cardboard Box Configuration
# ============================================================================
BOX_SIZE = 0.05  # 5cm cube
BOX_MASS = 0.1  # kg
BOX_FRICTION = 0.8
BOX_RESTITUTION = 0.05

# Corrugated Kraft Cardboard Colors
COLOR_CARDBOARD_KRAFT = [0.82, 0.67, 0.46, 1.0]  # Standard Kraft paper
COLOR_CARDBOARD_LIGHT = [0.86, 0.72, 0.52, 1.0]  # Light Kraft
COLOR_CARDBOARD_DARK = [0.74, 0.58, 0.38, 1.0]   # Heavy-duty Kraft

# Packaging tape accent colors for visual distinction
BOX_COLORS = {
    0: [0.82, 0.67, 0.46, 1.0],  # Kraft Cardboard #1
    1: [0.86, 0.72, 0.52, 1.0],  # Kraft Cardboard #2
    2: [0.76, 0.60, 0.40, 1.0],  # Kraft Cardboard #3
    3: [0.80, 0.64, 0.44, 1.0],  # Kraft Cardboard #4
    4: [0.84, 0.70, 0.50, 1.0],  # Kraft Cardboard #5
    5: [0.72, 0.56, 0.36, 1.0],  # Kraft Cardboard #6
}

BOX_TAPE_COLORS = {
    0: [0.85, 0.25, 0.20, 1.0],  # Red tape (Fragile / Electronics)
    1: [0.25, 0.65, 0.30, 1.0],  # Green tape (Standard / Eco)
    2: [0.25, 0.45, 0.85, 1.0],  # Blue tape (Priority / Express)
    3: [0.90, 0.75, 0.15, 1.0],  # Yellow tape (Heavy / Hardware)
    4: [0.85, 0.45, 0.10, 1.0],  # Orange tape (Hazard / Chemical)
    5: [0.60, 0.30, 0.70, 1.0],  # Purple tape (Special Handling)
}

# SKU inventory items for interactive box selection
BOX_CATALOG = [
    {"sku": "ELEC-101", "name": "Microcontroller Kits", "category": 0, "weight": "0.10 kg", "tape": "Red"},
    {"sku": "TOOL-202", "name": "Precision Hand Tools", "category": 1, "weight": "0.15 kg", "tape": "Green"},
    {"sku": "BOOK-303", "name": "Technical Manuals", "category": 2, "weight": "0.12 kg", "tape": "Blue"},
    {"sku": "AUTO-404", "name": "Robotic Sensors Pack", "category": 0, "weight": "0.08 kg", "tape": "Yellow"},
    {"sku": "PHAR-505", "name": "First Aid & Safety", "category": 1, "weight": "0.11 kg", "tape": "Orange"},
    {"sku": "PACK-606", "name": "Industrial Fasteners", "category": 2, "weight": "0.14 kg", "tape": "Purple"},
]

# ============================================================================
# Industrial Shelf Configuration
# ============================================================================
COLOR_INDUSTRIAL_RACK_UPRIGHT = [0.88, 0.42, 0.10, 1.0]  # Industrial Safety Orange
COLOR_INDUSTRIAL_RACK_BEAM = [0.88, 0.42, 0.10, 1.0]     # Matching beam orange
COLOR_INDUSTRIAL_SHELF_DECK = [0.72, 0.74, 0.76, 1.0]     # Galvanized steel wire deck

# Shelf dimensions (width x depth x height)
SHELF_WIDTH = 0.30  # meters (Y direction)
SHELF_DEPTH = 0.25  # meters (X direction) - shallow so the arm isn't blocked
SHELF_HEIGHT = 0.6  # meters (Z direction) - low so the arm sweeps clear above
SHELF_THICKNESS = 0.02  # meters

# Shelf positions (center of shelf base).
# The shelf rack sits BEHIND the robot base (x=0.70), so the arm is positioned
# BETWEEN the shelf and the bins.  The shelf at x=0.30 puts the boxes at
# x~0.42, only 0.28m behind the base - comfortably within the arm's 0.855m
# reach.  The bins sit IN FRONT of the robot (x=1.10, y=+/-0.30) so the arm
# reaches forward to place boxes.  The arm is literally in the middle.
SHELF_POSITIONS = [
    [0.30, -0.30, SHELF_HEIGHT / 2],  # Left shelf (rack of boxes, behind)
    [0.30, 0.30, SHELF_HEIGHT / 2],   # Right shelf (rack of boxes, behind)
]
# Shelf levels (height of each shelf board from ground)
# All levels are comfortably within the arm's vertical reach from the base at
# x=0.70 when reaching BACKWARD to the shelf at x=0.30.  Lowered so even the
# top shelf is easily reachable.
SHELF_LEVELS = [0.25, 0.40, 0.55]  # meters (all comfortably reachable)

# Number of boxes per shelf level
BOXES_PER_LEVEL = 2

# ============================================================================
# Bin Configuration
# ============================================================================
# Bin dimensions (inner dimensions)
BIN_WIDTH = 0.15  # meters (Y direction)
BIN_DEPTH = 0.15  # meters (X direction)
BIN_HEIGHT = 0.15  # meters (Z direction)
BIN_WALL_THICKNESS = 0.01  # meters

# Bin positions (center of bin base).
# The 3 bins sit IN FRONT of the robot base (x = 0.70): ~0.40 m forward,
# comfortably within the arm's 0.855 m reach.  The shelf is BEHIND the robot,
# so the arm reaches backward to grasp boxes and forward to place them in the
# bins - the arm is positioned BETWEEN the shelf and the bins.
BIN_POSITIONS = [
    [1.10, -0.30, BIN_HEIGHT / 2],   # Bin 0 (left, front)
    [1.10, 0.00, BIN_HEIGHT / 2],    # Bin 1 (center, front)
    [1.10, 0.30, BIN_HEIGHT / 2],    # Bin 2 (right, front)
]

# Bin colors (visual only)
BIN_COLORS = [
    [0.8, 0.2, 0.2, 0.7],  # Red transparent
    [0.2, 0.8, 0.2, 0.7],  # Green transparent
    [0.2, 0.2, 0.8, 0.7],  # Blue transparent
]

# ============================================================================
# Conveyor Belt Configuration
# ============================================================================
# The conveyor belt runs along the Y axis in front of the robot (X = 1.05).
# The robot reaches forward to place picked cardboard boxes onto the belt intake zone.
CONVEYOR_X = 1.05  # meters (X position in front of robot base at x=0.70)
CONVEYOR_Y_CENTER = 0.0  # meters (centered along Y axis)
CONVEYOR_LENGTH = 1.30  # meters along Y axis (from y=-0.65 to y=+0.65)
CONVEYOR_WIDTH = 0.24  # meters along X axis (from x=0.93 to x=1.17)
CONVEYOR_HEIGHT = 0.22  # meters (height of conveyor belt surface from ground)
CONVEYOR_BED_THICKNESS = 0.04  # meters
CONVEYOR_RAIL_HEIGHT = 0.05  # meters (side guide rail height)
CONVEYOR_SPEED = 0.25  # m/s (conveyor belt speed along +Y direction)
CONVEYOR_DISPATCH_Y = 0.48  # Y coordinate threshold where box is considered delivered

# Drop position on conveyor belt (intake loading zone)
CONVEYOR_DROP_POSITION = [CONVEYOR_X, 0.0, CONVEYOR_HEIGHT + BOX_SIZE / 2 + 0.005]

# Conveyor visual colors
COLOR_CONVEYOR_BELT = [0.18, 0.18, 0.20, 1.0]   # Matte charcoal rubber
COLOR_CONVEYOR_RAIL = [0.88, 0.68, 0.12, 1.0]   # High-vis safety yellow
COLOR_CONVEYOR_LEGS = [0.25, 0.26, 0.28, 1.0]   # Industrial steel legs
COLOR_CONVEYOR_TOTE = [0.20, 0.45, 0.75, 0.9]   # Blue dispatch container

# ============================================================================
# Franka Panda Robot Configuration
# ============================================================================
# Robot base position.
# Layout (X axis):
#   Shelf rack:          x = 0.30, y = +/-0.30   (BEHIND)
#   ROBOT BASE:          x = 0.70              (arm in the MIDDLE)
#   Bins:                x = 1.10, y = +/-0.30   (IN FRONT)
# The arm is positioned BETWEEN the shelf and the bins.  It reaches BACKWARD
# to grasp shelf boxes (0.28m behind the base - easily reachable), then
# FORWARD to place them into the bins.  This is the natural layout the user
# requested.
ROBOT_BASE_POSITION = [0.70, 0, 0]  # meters

# Joint limits (from Franka Panda datasheet)
# [lower, upper] in radians
FRANKA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # Joint 1
    (-1.7628, 1.7628),   # Joint 2
    (-2.8973, 2.8973),   # Joint 3
    (-3.0718, -0.0698),  # Joint 4
    (-2.8973, 2.8973),   # Joint 5
    (-0.0175, 3.7525),   # Joint 6
    (-2.8973, 2.8973),   # Joint 7
]

# Joint velocities (rad/s)
FRANKA_JOINT_VELOCITIES = [2.1750] * 7

# End effector link name.
# The Panda URDF has TWO candidate end-effector links:
#   - "panda_hand"        (link 8)  - the hand body, ~65mm ABOVE the grasp point
#   - "panda_grasptarget" (link 11) - the actual point where objects are grasped
# Using "panda_hand" as the IK target introduces a systematic ~40mm offset,
# because the hand body sits above the real grasp point.  Targeting the grasp
# target link makes the IK reach the actual box position, which is the single
# biggest factor in the pick success rate.
FRANKA_EE_LINK_NAME = "panda_grasptarget"
FRANKA_GRASP_TARGET_LINK_NAME = "panda_grasptarget"
FRANKA_HAND_LINK_NAME = "panda_hand"

# Gripper finger joint names.
# The standard Franka Panda URDF exposes exactly two prismatic finger joints
# (panda_finger_joint1 / panda_finger_joint2).  We detect gripper joints
# DYNAMICALLY in robot.py (any joint whose name contains "finger" or whose
# type is prismatic), so these names are only used as a fallback.  Keeping the
# extra "joint3/joint4" entries does no harm: they simply won't exist in the
# URDF and are skipped.
FRANKA_GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_finger_joint3",
    "panda_finger_joint4",
]

# Gripper limits
# The Franka Panda gripper fingers are prismatic joints with range [0, 0.04].
#   GRIPPER_OPEN_POS  = 0.04  -> fully open (0.08 m total finger gap)
#   GRIPPER_CLOSE_POS = 0.00  -> fully closed (fingers touching)
# IMPORTANT: A 5cm box means each finger rests at ~0.025 when closed around
# it, so grasp detection must treat "near 0.025" as closed, NOT require
# fingers to reach 0.02 (which is impossible around a 5cm box).
GRIPPER_OPEN_POS = 0.04  # meters (fully open)
GRIPPER_CLOSE_POS = 0.00  # meters (fully closed)
GRIPPER_FORCE = 80.0  # Newtons (raised from 50 for a firmer, more reliable grip)

# Gripper max velocity (m/s).  The stock value of 0.03 was far too slow for the
# fingers to reach their target within the grasp sequence, so the gripper never
# opened/closed fully.  Raising it lets the fingers "move freely" and reach
# their commanded position reliably.
GRIPPER_MAX_VELOCITY = 0.2  # m/s

# Whether the gripper control should move all finger joints simultaneously.
# Together with GRIPPER_MAX_VELOCITY this makes every gripper joint move
# freely and in step, rather than one at a time.
GRIPPER_SIMULTANEOUS = True
# The expected resting finger position when closed around a box of BOX_SIZE.
# Each finger stops at half the box size (0.05/2 = 0.025). Used by the grasp
# detector to decide whether the fingers are "closed enough" around a box.
GRIPPER_EXPECTED_CLOSED_POS = BOX_SIZE / 2  # 0.025 m
# ============================================================================
# Grasp Detection
# ============================================================================
# Minimum contact force to consider a grasp successful (Newtons)
# Lowered from 5.0 so a light but real contact counts as a grasp.  The
# fingers press the box with GRIPPER_FORCE (80N), so any real contact will
# exceed this easily; a lower threshold just avoids false negatives.
MIN_GRASP_FORCE = 3.0

# Maximum gripper opening to consider "closed" (meters)
# Relaxed from 0.02 to 0.03: a 5cm box keeps the fingers at ~0.025 each, so
# requiring < 0.02 made physical grasp detection impossible.  With 0.03 the
# fingers resting around the box count as "closed".
MAX_GRIPPER_OPEN_FOR_GRASP = 0.03
# Tolerance (meters) around GRIPPER_EXPECTED_CLOSED_POS within which the
# fingers are considered to be gripping a box.  A 5cm box -> fingers at
# 0.025 +/- 0.015.
GRIPPER_CLOSE_TOLERANCE = 0.015

# ============================================================================
# Motion Primitive Parameters
# ============================================================================
# Approach height above object (meters)
APPROACH_HEIGHT = 0.15

# Lift height after grasp (meters)
LIFT_HEIGHT = 0.15

# Descent speed (m/s)
DESCENT_SPEED = 0.05

# Movement speed (m/s)
MOVE_SPEED = 0.2

# IK solver parameters
IK_POSITION_TOLERANCE = 0.001  # meters
IK_ORIENTATION_TOLERANCE = 0.01  # radians
IK_MAX_ITERATIONS = 100

# ============================================================================
# Reward Signal Thresholds (for RL team reference)
# ============================================================================
REWARD_GRASP_DISTANCE_THRESHOLD = 0.1  # meters
REWARD_PLACE_DISTANCE_THRESHOLD = 0.1  # meters

# ============================================================================
# Episode Configuration
# ============================================================================
MAX_EPISODE_STEPS = 500
MAX_GRASP_RETRIES = 3

# ============================================================================
# Helper Functions
# ============================================================================


# Custom layout overrides (when user modifies layout.json)
CUSTOM_BOX_SPAWNS = None


def apply_layout(layout_dict: dict):
    """
    Apply layout positions for robot, conveyor, shelves, and boxes at will.
    """
    global ROBOT_BASE_POSITION, CONVEYOR_X, CONVEYOR_Y_CENTER, CONVEYOR_HEIGHT
    global CONVEYOR_LENGTH, CONVEYOR_WIDTH, CONVEYOR_SPEED, CONVEYOR_DROP_POSITION
    global SHELF_POSITIONS, SHELF_LEVELS, CUSTOM_BOX_SPAWNS

    if "arm" in layout_dict and "base_position" in layout_dict["arm"]:
        ROBOT_BASE_POSITION = list(layout_dict["arm"]["base_position"])

    if "conveyor" in layout_dict:
        c = layout_dict["conveyor"]
        pos = c.get("position", [1.05, 0.0, 0.22])
        CONVEYOR_X = float(pos[0])
        CONVEYOR_Y_CENTER = float(pos[1])
        CONVEYOR_HEIGHT = float(pos[2])
        CONVEYOR_LENGTH = float(c.get("length", 1.30))
        CONVEYOR_WIDTH = float(c.get("width", 0.24))
        CONVEYOR_SPEED = float(c.get("speed", 0.25))
        CONVEYOR_DROP_POSITION = [CONVEYOR_X, CONVEYOR_Y_CENTER, CONVEYOR_HEIGHT + BOX_SIZE / 2 + 0.005]

    if "shelves" in layout_dict and layout_dict["shelves"]:
        SHELF_POSITIONS = [list(s["position"]) for s in layout_dict["shelves"]]
        if "levels" in layout_dict["shelves"][0]:
            SHELF_LEVELS = list(layout_dict["shelves"][0]["levels"])

    if "boxes" in layout_dict and layout_dict["boxes"]:
        CUSTOM_BOX_SPAWNS = [list(b["spawn_position"]) for b in layout_dict["boxes"]]


def get_box_spawn_positions():
    """
    Generate valid spawn positions for boxes on shelves.
    Uses custom positions if defined in layout, otherwise calculates from shelf positions.
    """
    if CUSTOM_BOX_SPAWNS is not None and len(CUSTOM_BOX_SPAWNS) > 0:
        return [list(p) for p in CUSTOM_BOX_SPAWNS]

    positions = []
    for shelf_pos in SHELF_POSITIONS:
        for level in SHELF_LEVELS:
            for i in range(BOXES_PER_LEVEL):
                y_offset = (i - (BOXES_PER_LEVEL - 1) / 2) * 0.12
                x = shelf_pos[0] + SHELF_DEPTH / 2 - BOX_SIZE / 2 - 0.01
                y = shelf_pos[1] + y_offset
                z = level + BOX_SIZE / 2 + 0.01
                positions.append([x, y, z])
    return positions


def get_bin_for_category(category):
    """
    Map a box category to a bin index.
    
    Args:
        category (int): Box category (0, 1, or 2)
    
    Returns:
        int: Bin index
    """
    return category % len(BIN_POSITIONS)


def get_box_color(category):
    """
    Get RGBA color for a box based on its category.
    
    Args:
        category (int): Box category
    
    Returns:
        list: RGBA color values [r, g, b, a]
    """
    return BOX_COLORS.get(category % len(BOX_COLORS), COLOR_CARDBOARD_KRAFT)


def get_box_tape_color(category):
    """
    Get RGBA color for packaging tape on top of cardboard box.
    """
    return BOX_TAPE_COLORS.get(category % len(BOX_TAPE_COLORS), [0.85, 0.25, 0.20, 1.0])


def get_box_catalog_item(index):
    """
    Get SKU catalog item information for a box by index.
    """
    return BOX_CATALOG[index % len(BOX_CATALOG)]
