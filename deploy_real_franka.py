
import sys
import time
import argparse
import json
import numpy as np

try:
    from frankapy import FrankaArm
    HAS_FRANKAPY = True
except ImportError:
    HAS_FRANKAPY = False


def execute_on_hardware(trajectory_file: str, robot_ip: str, execution_speed: float = 1.0):
    print("=" * 65)
    print("          DEPLOYING TRAJECTORY TO PHYSICAL FRANKA PANDA       ")
    print("=" * 65)
    print(f"Target Robot IP: {robot_ip}")
    print(f"Trajectory File: {trajectory_file}")
    print(f"Speed Scaling:   {execution_speed}x")

    with open(trajectory_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actions = data.get("actions", [])
    print(f"Loaded {len(actions)} action sequences.")

    if not HAS_FRANKAPY:
        print("\n[SIMULATED HARDWARE RUN]")
        print("Note: 'frankapy' is not installed on this host.")
        print("This script provides the exact hardware command structure.")
        print("To run on the physical Franka arm, install frankapy (pip install frankapy)\n")

        for act_idx, action in enumerate(actions):
            print(f"--- Action {act_idx + 1}: {action.get('action')} ---")
            waypoints = action.get("waypoints", [])
            print(f"Executing {len(waypoints)} joint waypoints on hardware...")
            for i in range(0, len(waypoints), max(1, len(waypoints) // 6)):
                pt = waypoints[i]
                print(f"  t={pt['time']:.2f}s -> q={np.round(pt['joints'], 3)} | Gripper={pt['gripper']}m")
                time.sleep(0.1 / execution_speed)
        print("\n[OK] Hardware execution simulation finished.")
        return

    # Physical hardware execution via FrankaArm
    print("\nConnecting to Franka Control Interface (FCI)...")
    fa = FrankaArm()
    fa.reset_joints()
    time.sleep(1.0)

    for action in actions:
        waypoints = action.get("waypoints", [])
        if not waypoints:
            continue

        for pt in waypoints:
            target_q = np.array(pt["joints"])
            fa.goto_joints(target_q, duration=max(0.1, 0.05 / execution_speed), ignore_virtual_walls=True)

            # Gripper actuation
            if pt["gripper"] > 0.03:
                fa.open_gripper()
            else:
                fa.close_gripper()

    fa.reset_joints()
    print("\n[OK] Real robot pick-and-place completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute trajectory on real Franka Panda")
    parser.add_argument("--robot-ip", type=str, default="172.16.0.2", help="Franka robot IP")
    parser.add_argument("--trajectory", type=str, default="exported_trajectory.json", help="Trajectory JSON file")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    args = parser.parse_args()

    execute_on_hardware(args.trajectory, args.robot_ip, args.speed)
