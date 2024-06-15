#include "gui.h"

#include <iostream>

GuiRenderer::GuiRenderer(GLFWwindow *w, ImGui_ImplVulkan_InitInfo &imguiInitInfo,
                         const vk::raii::RenderPass &renderPass) : window(w) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);

    ImGui_ImplVulkan_Init(&imguiInitInfo, static_cast<VkRenderPass>(*renderPass));
}

GuiRenderer::~GuiRenderer() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GuiRenderer::startRendering() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                                       | ImGuiWindowFlags_NoCollapse
                                       | ImGuiWindowFlags_NoSavedSettings
                                       | ImGuiWindowFlags_NoResize
                                       | ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(0, 0));

    glm::ivec2 windowSize;
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);
    ImGui::SetNextWindowSize(ImVec2(0, windowSize.y));

    ImGui::Begin("main window", nullptr, flags);
}

void GuiRenderer::finishRendering(const vk::raii::CommandBuffer &commandBuffer) {
    ImGui::End();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandBuffer);
}
