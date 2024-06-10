
@echo off
for /D %%i in (C:\VulkanSDK\*) do set "SDK_DIR=%%i"
set "IS_ERROR=0"

@echo on
%SDK_DIR%/Bin/glslc.exe shader.vert -o vert.spv -g --target-env=vulkan1.1
@echo off
if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"

@echo on
%SDK_DIR%/Bin/glslc.exe shader.frag -o frag.spv -g --target-env=vulkan1.1
@echo off
if %ERRORLEVEL% NEQ 0 set "IS_ERROR=1"

if %IS_ERROR% NEQ 0 exit 1
