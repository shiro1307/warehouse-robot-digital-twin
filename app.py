"""
RoboWarehouse Studio & Game - Desktop Application Launcher.

Launch by running:
    python app.py
or double-clicking 'start_app.bat' in Windows.
"""

import sys
import os

# Configure High-Performance Dedicated GPU (NVIDIA RTX 2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SHIM_MCCOMPAT"] = "0x800000001"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

if sys.platform == "win32":
    try:
        import ctypes
        _dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_enable.dll")
        if os.path.exists(_dll_path):
            ctypes.CDLL(_dll_path)
    except Exception:
        pass

from warehouse_sim.app import main

if __name__ == "__main__":
    main()

