# PBR Model Viewer

A simple app for viewing 3D models, powered by Vulkan.
Rendering uses Physically-Based Rendering (PBR) techniques, incorporating Image Based Lighting (IBL) with HDR environment maps to achieve realistic lighting.

### Features

* Loading all kinds of models using Assimp
* PBR lighting model using the Cook-Torrance GGX
* IBL using user selectable HDR environment maps, convolved and prefiltered at runtime
* SSAO usable interchangably with baked AO maps provided during model loading
* Instancing used to minimize draw calls
* ImGui user interface

### Compilation

To compile this code you need to have the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) installed.

The project is compiled simply using `cmake`:
```
cmake --build <build-directory-name>
```

When modifying shaders, they need to be recompiled. To compile shaders, first go into the `shaders` directory and run:
```
./compile.bat
```

Then, run an exporter tool (which is compiled alongside the main executable) that exports shader Spir-V code into an embeddable C++ source file:
```
cd <build-directory-name>
shader_export.exe
```

Keep in mind that currently this project requires your machine to support Vulkan 1.3 (which is subject for change in the future).
All other dependencies are included in the `deps` subdirectory.

### Controls

Press `` ` `` to open/close the GUI.
Drag the left mouse button to rotate the camera around your model. 
Drag the right mouse button to pan your model.

### Gallery

![image](https://github.com/user-attachments/assets/9a6a3a0f-7e76-4f68-8481-ecd6b327f0ce)
![image](https://github.com/user-attachments/assets/4b7b9a92-a21f-4458-a53e-a2d1ef64e6a1)
![image](https://github.com/user-attachments/assets/6c6d35d7-b13b-42ea-8006-2b8db69050bf)
![image](https://github.com/user-attachments/assets/20b304db-8cd0-4574-9e64-9c5c42f2ce31)

### Future improvements

* Add more post-processing effects
* More lighting customization (add point lights)
* Maybe add shadows
* Add a blur option for the background environment
* 
* Clean up code
