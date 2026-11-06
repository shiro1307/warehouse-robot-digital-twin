"""
MuJoCo Warehouse Fulfillment Simulation Launcher.

Runs the Franka Emika Panda warehouse simulation using Google DeepMind's
MuJoCo physics and PBR rendering engine.

Usage:
    python -m warehouse_sim.mujoco_main                  # Interactive box selection
    python -m warehouse_sim.mujoco_main --box 1          # Pick box #1
    python -m warehouse_sim.mujoco_main --box ELEC-101   # Pick by SKU
    python -m warehouse_sim.mujoco_main --all            # Automated full episode
    python -m warehouse_sim.mujoco_main --export-trajectory # Export for real robot arm
"""

import sys
import os
import time
import argparse
import numpy as np

from warehouse_sim.mujoco_env import MuJoCoWarehouseEnv
from warehouse_sim.layout_manager import LayoutManager
from warehouse_sim import config


def print_banner():
    """Print ASCII banner for MuJoCo simulation."""
    print("=" * 64)
    print("  ROBOWAREHOUSE - GOOGLE DEEPMIND MUJOCO 3D DIGITAL TWIN")
    print("  Franka Emika Panda 7-DOF Cobot | Industrial Pallet Racks")
    print("  Photorealistic PBR Rendering & Contact Physics")
    print("=" * 64)


def display_inventory(env: MuJoCoWarehouseEnv):
    """Display cardboard box inventory catalog."""
    boxes = env.layout_data.get("boxes", [])
    print("\n[INVENTORY] Cardboard Boxes on Industrial Shelf:")
    print(f"  {'#':<4} {'SKU':<10} {'Item Name':<24} {'Status':<16}")
    print("  " + "-" * 56)
    for i, b in enumerate(boxes):
        status = env.box_status.get(b['sku'], 'ON_SHELF')
        status_str = "[OK] ON SHELF" if status == "ON_SHELF" else "[>] ON CONVEYOR"
        print(f"  [{i+1}]  {b['sku']:<10} {b.get('name', 'Box'):<24} {status_str}")
    print()


def run_interactive(env: MuJoCoWarehouseEnv, export_traj: bool = False):
    """Run interactive terminal menu allowing user to choose any box."""
    import threading
    boxes = env.layout_data.get("boxes", [])
    running = True
    busy_lock = threading.Lock()

    def idle_stepper():
        while running:
            if not busy_lock.locked():
                try:
                    env.step_simulation(1)
                except Exception:
                    pass
            time.sleep(1.0 / 60.0)

    stepper_thread = threading.Thread(target=idle_stepper, daemon=True)
    stepper_thread.start()

    try:
        while True:
            display_inventory(env)
            remaining = [b for b in boxes if env.box_status.get(b['sku']) == "ON_SHELF"]
            if not remaining:
                print("[CONGRATULATIONS] All 6 boxes have been successfully dispatched onto the conveyor belt!")
                break

            print("[MENU] Choose which box to pick:")
            print("  * Enter box number [1-6] to pick that specific box")
            print("  * Enter 'A' to dispatch ALL remaining boxes in sequence")
            print("  * Enter 'R' to dispatch a random available box")
            print("  * Enter 'Q' to quit")

            try:
                choice = input("\nYour choice > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting simulation.")
                break

            if not choice:
                continue
            if choice.upper() == "Q":
                print("Exiting.")
                break
            elif choice.upper() == "A":
                print("\n[AUTO] Dispatching all remaining boxes...")
                with busy_lock:
                    for b in remaining:
                        sku = b["sku"]
                        print(f"\n--> Picking box {sku} ({b.get('name', '')})...")
                        ok, msg = env.pick_box_to_conveyor(sku)
                        print(f"    {msg}")
                break
            elif choice.upper() == "R":
                target = np.random.choice(remaining)
                sku = target["sku"]
                print(f"\n--> Picking random box: {sku} ({target.get('name', '')})...")
                with busy_lock:
                    ok, msg = env.pick_box_to_conveyor(sku)
                print(f"    {msg}")
            else:
                # Check if number
                target_sku = None
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(boxes):
                        target_sku = boxes[idx]["sku"]
                else:
                    for b in boxes:
                        if b["sku"].upper() == choice.upper():
                            target_sku = b["sku"]
                            break

                if target_sku is None:
                    print(f"[ERROR] Invalid choice '{choice}'. Enter 1-6 or SKU.")
                    continue

                if env.box_status.get(target_sku) != "ON_SHELF":
                    print(f"[WARN] Box {target_sku} is already dispatched!")
                    continue

                print(f"\n--> Picking chosen box: {target_sku}...")
                with busy_lock:
                    ok, msg = env.pick_box_to_conveyor(target_sku)
                print(f"    {msg}")
    finally:
        running = False
        stepper_thread.join(timeout=1.0)

    if export_traj:
        env.hardware.export_trajectory_json("exported_trajectory.json")
        env.hardware.export_trajectory_csv("exported_trajectory.csv")
        env.hardware.generate_real_franka_script("deploy_real_franka.py")
        print("\n[OK] Hardware trajectories exported for real Franka Panda arm.")


def main():
    parser = argparse.ArgumentParser(description="Franka Panda MuJoCo Warehouse Simulation")
    parser.add_argument("--box", type=str, default=None, help="Choose specific box (1-6 or SKU like ELEC-101)")
    parser.add_argument("--all", action="store_true", help="Automated dispatch of all boxes")
    parser.add_argument("--preset", type=str, choices=["standard", "side_conveyor", "compact"], default=None)
    parser.add_argument("--layout", type=str, default="layout.json", help="Path to layout.json")
    parser.add_argument("--export-trajectory", action="store_true", help="Export joint waypoints for real arm")
    parser.add_argument("--headless", action="store_true", help="Run without 3D GUI window")
    args = parser.parse_args()

    print_banner()

    # Apply layout preset if specified
    if args.preset:
        preset = LayoutManager.get_preset(args.preset)
        LayoutManager.save_layout(preset, args.layout)
        print(f"[OK] Applied preset: '{args.preset}'")

    env = MuJoCoWarehouseEnv(gui=not args.headless, layout_file=args.layout)

    try:
        if args.all:
            print("[AUTO] Dispatching all boxes...")
            stats = env.execute_scripted_episode()
            print(f"[COMPLETE] Placed {stats['successful_placements']}/{stats['total_boxes']} boxes onto conveyor.")
            if args.export_trajectory:
                env.hardware.export_trajectory_json("exported_trajectory.json")
                env.hardware.export_trajectory_csv("exported_trajectory.csv")
                env.hardware.generate_real_franka_script("deploy_real_franka.py")
            time.sleep(2.0)
        elif args.box:
            # Pick single box
            boxes = env.layout_data.get("boxes", [])
            target_sku = None
            if args.box.isdigit():
                idx = int(args.box) - 1
                if 0 <= idx < len(boxes):
                    target_sku = boxes[idx]["sku"]
            else:
                for b in boxes:
                    if b["sku"].upper() == args.box.upper():
                        target_sku = b["sku"]
                        break
            if not target_sku:
                target_sku = "ELEC-101"
            print(f"[ACTION] Picking box {target_sku}...")
            ok, msg = env.pick_box_to_conveyor(target_sku)
            print(f"Result: {msg}")
            if args.export_trajectory:
                env.hardware.export_trajectory_json("exported_trajectory.json")
                env.hardware.export_trajectory_csv("exported_trajectory.csv")
                env.hardware.generate_real_franka_script("deploy_real_franka.py")
            time.sleep(2.0)
        else:
            run_interactive(env, export_traj=args.export_trajectory)
    finally:
        env.close()


if __name__ == "__main__":
    main()

