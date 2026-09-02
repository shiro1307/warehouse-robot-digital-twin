import mujoco
import mujoco.viewer
import time
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "mujoco_menagerie",
    "franka_emika_panda",
    "scene.xml"
)


def main():
    print("=" * 50)
    print("Franka Panda - MuJoCo")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Panda model not found:")
        print(MODEL_PATH)
        return

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("\nModel loaded!")
    print(f"Joints: {model.njnt}")
    print(f"DOFs:   {model.nv}")
    print(f"Bodies: {model.nbody}")
    print(f"Geoms:  {model.ngeom}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nSimulation running.")
        print("Close the MuJoCo window to exit.")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()