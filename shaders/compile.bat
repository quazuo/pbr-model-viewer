
@echo off
for /D %%i in (C:\VulkanSDK\*) do set "SDK_DIR=%%i"
set "IS_ERROR=0"

set shaders="shader" "skybox" "sphere-cube" "convolute"

(for %%a in (%shaders%) do (
   @echo on
   %SDK_DIR%/Bin/glslc.exe %%a.vert -o obj/%%a-vert.spv -g --target-env=vulkan1.1
   @echo off
   if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"

   @echo on
   %SDK_DIR%/Bin/glslc.exe %%a.frag -o obj/%%a-frag.spv -g --target-env=vulkan1.1
   @echo off
   if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"
))

if %IS_ERROR% NEQ 0 exit 1
