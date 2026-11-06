"""
Warehouse Layout Manager.

Allows configuring, saving, loading, and validating custom factory layouts
where the Franka Panda arm, conveyor belt, industrial shelves, and boxes
can be positioned according to will.
"""

import json
import os
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


DEFAULT_LAYOUT_FILE = "layout.json"


class LayoutManager:
    """
    Manages custom layout configurations for the warehouse environment.
    """

    @staticmethod
    def get_default_layout() -> Dict:
        """Return the default factory layout dictionary."""
        return {
            "version": "1.0",
            "description": "Industrial Logistics Cell: Franka Panda Arm with Pallet Racks and Conveyor Belt",
            "arm": {
                "base_position": [0.70, 0.0, 0.0],
                "model": "Franka Emika Panda 7-DOF"
            },
            "conveyor": {
                "position": [1.05, 0.0, 0.22],
                "length": 1.30,
                "width": 0.24,
                "height": 0.22,
                "speed": 0.25
            },
            "shelves": [
                {
                    "name": "Left Shelf Rack",
                    "position": [0.30, -0.30, 0.30],
                    "levels": [0.25, 0.40, 0.55]
                },
                {
                    "name": "Right Shelf Rack",
                    "position": [0.30, 0.30, 0.30],
                    "levels": [0.25, 0.40, 0.55]
                }
            ],
            "boxes": [
                {"sku": "TOOL-202", "name": "Precision Hand Tools", "spawn_position": [0.38, -0.18, 0.28], "category": 1},
                {"sku": "PHAR-505", "name": "First Aid & Safety", "spawn_position": [0.38, -0.42, 0.43], "category": 1},
                {"sku": "BOOK-303", "name": "Technical Manuals", "spawn_position": [0.38, -0.18, 0.58], "category": 2},
                {"sku": "AUTO-404", "name": "Robotic Sensors Pack", "spawn_position": [0.38, 0.18, 0.28], "category": 0},
                {"sku": "ELEC-101", "name": "Microcontroller Kits", "spawn_position": [0.38, 0.42, 0.43], "category": 0},
                {"sku": "PACK-606", "name": "Industrial Fasteners", "spawn_position": [0.38, 0.18, 0.58], "category": 2}
            ]
        }

    @staticmethod
    def load_layout(filepath: str = DEFAULT_LAYOUT_FILE) -> Dict:
        """
        Load factory layout from JSON file. If file does not exist,
        creates the default layout file.
        """
        if not os.path.exists(filepath):
            default_data = LayoutManager.get_default_layout()
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=2)
            except Exception:
                pass
            return default_data

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"[WARN] Failed to load layout from '{filepath}': {e}. Using default layout.")
            return LayoutManager.get_default_layout()

    @staticmethod
    def save_layout(layout_data: Dict, filepath: str = DEFAULT_LAYOUT_FILE):
        """Save factory layout configuration to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2)
        print(f"[OK] Layout saved to '{filepath}'")

    @staticmethod
    def validate(layout_data: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        Validate layout reachability. If layout_data is None, validates current loaded layout.
        Returns: (is_valid, list_of_warnings)
        """
        if layout_data is None:
            layout_data = LayoutManager.load_layout()
        warnings = LayoutManager.validate_reachability(layout_data)
        return len(warnings) == 0, warnings

    @staticmethod
    def validate_reachability(layout_data: Dict) -> List[str]:
        """
        Validate whether placed shelves, conveyor, and boxes are within the
        Franka Panda arm's physical working envelope (0.25m <= distance <= 0.855m).
        
        Returns:
            list: List of warning messages for unreachable or questionable placements.
        """
        warnings = []
        arm_base = np.array(layout_data.get("arm", {}).get("base_position", [0.70, 0.0, 0.0]))
        max_reach = 0.92  # Franka Panda 0.855m bare arm + 0.13m parallel gripper envelope
        min_reach = 0.22

        # Check conveyor drop point
        conv_pos = np.array(layout_data.get("conveyor", {}).get("position", [1.05, 0.0, 0.22]))
        conv_dist = float(np.linalg.norm(conv_pos[:2] - arm_base[:2]))
        if conv_dist > max_reach:
            warnings.append(f"Conveyor at {conv_pos[:2]} is {conv_dist:.2f}m from robot base (exceeds max reach {max_reach}m)")
        elif conv_dist < min_reach:
            warnings.append(f"Conveyor at {conv_pos[:2]} is {conv_dist:.2f}m from robot base (too close to base column)")

        # Check boxes
        boxes = layout_data.get("boxes", [])
        for box in boxes:
            pos = np.array(box.get("spawn_position", [0, 0, 0]))
            dist = float(np.linalg.norm(pos - arm_base))
            sku = box.get("sku", "Box")
            if dist > max_reach:
                warnings.append(f"Box {sku} at {pos} is {dist:.2f}m from arm base (out of reach: max {max_reach}m)")
            elif dist < min_reach:
                warnings.append(f"Box {sku} at {pos} is {dist:.2f}m from arm base (too close to base singularity)")

        return warnings

    @staticmethod
    def get_preset(name: str) -> Dict:
        """
        Get a predefined warehouse layout preset.
        
        Presets:
        - 'standard': Arm in middle (x=0.70), conveyor in front (x=1.05), shelves behind (x=0.30).
        - 'side_conveyor': 90-degree transfer cell (shelf on left, conveyor on right).
        - 'inline': Straight-line packaging cell.
        """
        preset = LayoutManager.get_default_layout()

        if name == "side_conveyor":
            preset["description"] = "90-Degree Workcell: Left Shelf to Right Conveyor"
            preset["arm"]["base_position"] = [0.50, 0.0, 0.0]
            # Conveyor on right side (along X axis)
            preset["conveyor"]["position"] = [0.50, 0.40, 0.22]
            # Shelves on left side (Y = -0.40)
            preset["shelves"] = [
                {"name": "Left Rack 1", "position": [0.40, -0.38, 0.30], "levels": [0.25, 0.40, 0.55]},
                {"name": "Left Rack 2", "position": [0.60, -0.38, 0.30], "levels": [0.25, 0.40, 0.55]}
            ]
            preset["boxes"] = [
                {"sku": "ELEC-101", "name": "Microcontrollers", "spawn_position": [0.40, -0.32, 0.58], "category": 0},
                {"sku": "TOOL-202", "name": "Hand Tools", "spawn_position": [0.40, -0.32, 0.43], "category": 1},
                {"sku": "BOOK-303", "name": "Manuals", "spawn_position": [0.40, -0.32, 0.28], "category": 2},
                {"sku": "AUTO-404", "name": "Sensors", "spawn_position": [0.60, -0.32, 0.58], "category": 0},
                {"sku": "PHAR-505", "name": "First Aid", "spawn_position": [0.60, -0.32, 0.43], "category": 1},
                {"sku": "PACK-606", "name": "Fasteners", "spawn_position": [0.60, -0.32, 0.28], "category": 2}
            ]

        elif name == "compact":
            preset["description"] = "Compact Industrial Cell: Tight reach for rapid cycle time"
            preset["arm"]["base_position"] = [0.60, 0.0, 0.0]
            preset["conveyor"]["position"] = [0.90, 0.0, 0.22]
            preset["shelves"] = [
                {"name": "Rear Shelf", "position": [0.35, 0.0, 0.30], "levels": [0.25, 0.40, 0.55]}
            ]
            preset["boxes"] = [
                {"sku": "ELEC-101", "name": "Microcontrollers", "spawn_position": [0.35, -0.06, 0.58], "category": 0},
                {"sku": "TOOL-202", "name": "Hand Tools", "spawn_position": [0.35, 0.06, 0.58], "category": 1},
                {"sku": "BOOK-303", "name": "Manuals", "spawn_position": [0.35, -0.06, 0.43], "category": 2},
                {"sku": "AUTO-404", "name": "Sensors", "spawn_position": [0.35, 0.06, 0.43], "category": 0},
                {"sku": "PHAR-505", "name": "First Aid", "spawn_position": [0.35, -0.06, 0.28], "category": 1},
                {"sku": "PACK-606", "name": "Fasteners", "spawn_position": [0.35, 0.06, 0.28], "category": 2}
            ]

        return preset
