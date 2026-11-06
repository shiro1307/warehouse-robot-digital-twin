@echo off
title RoboWarehouse Studio & Game - Franka Digital Twin
echo Configuring High-Performance Dedicated GPU (NVIDIA RTX 2050)...
reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "C:\Users\abhis\AppData\Local\Programs\Python\Python311\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
set CUDA_VISIBLE_DEVICES=0
set SHIM_MCCOMPAT=0x800000001
set __NV_PRIME_RENDER_OFFLOAD=1
set __GLX_VENDOR_LIBRARY_NAME=nvidia
echo Launching RoboWarehouse Studio at Maximum GPU FPS...
python app.py
pause

