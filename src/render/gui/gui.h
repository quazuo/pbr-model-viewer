#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/render/libs.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#define NOMINMAX 1
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <imgui-filebrowser/imfilebrowser.h>
#include <imGuIZMO.quat/imGuIZMOquat.h>

class GuiRenderer {
    GLFWwindow *window;

public:
    explicit GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imguiInitInfo);

    ~GuiRenderer();

    GuiRenderer(const GuiRenderer& other) = delete;

    GuiRenderer& operator=(const GuiRenderer& other) = delete;

    void beginRendering();

    void endRendering(const vk::raii::CommandBuffer& commandBuffer);
};
