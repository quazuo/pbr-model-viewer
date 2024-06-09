#pragma once

#define IMGUI_DEFINE_MATH_OPERATORS

#include "GLFW/glfw3.h"
#include "deps/imgui/imgui.h"
#include "deps/imgui/backends/imgui_impl_glfw.h"
#include "deps/imgui/backends/imgui_impl_vulkan.h"

#include "../libs.h"

class GuiRenderer {
    GLFWwindow *window;

public:
    explicit GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imguiInitInfo,
                         const vk::raii::RenderPass &renderPass);

    ~GuiRenderer();

    GuiRenderer(const GuiRenderer& other) = delete;

    GuiRenderer& operator=(const GuiRenderer& other) = delete;

    void startRendering();

    void finishRendering(const vk::raii::CommandBuffer& commandBuffer);
};
