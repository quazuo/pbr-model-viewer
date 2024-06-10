#include <iostream>

#include "render/renderer.h"
#include "utils/key-manager.h"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    std::unique_ptr<KeyManager> keyManager;

    float lastTime = 0.0f;

    bool isCursorLocked = true;
    bool doShowGui = false;

public:
    Engine() {
        window = renderer.getWindow();

        renderer.setIsCursorLocked(isCursorLocked);
        renderer.setDoShowGui(doShowGui);

        keyManager = std::make_unique<KeyManager>(window);
        bindKeyActions();
    }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            tick();
        }

        renderer.waitIdle();
    }

private:
    void tick() {
        const auto currentTime = static_cast<float>(glfwGetTime());
        const float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        keyManager->tick(deltaTime);
        renderer.tick(deltaTime);

        renderer.startFrame();

        renderer.renderGui([&] {
            renderGuiSection(deltaTime);
            renderer.renderGuiSection();
        });

        renderer.drawScene();

        renderer.endFrame();
    }

    void bindKeyActions() {
        keyManager->bindCallback(GLFW_KEY_ESCAPE, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
            (void) deltaTime;
            glfwSetWindowShouldClose(window, true);
        });

        keyManager->bindCallback(GLFW_KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
            (void) deltaTime;
            doShowGui = !doShowGui;
            renderer.setDoShowGui(doShowGui);
        });

        keyManager->bindCallback(GLFW_KEY_F1, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
            (void) deltaTime;
            isCursorLocked = !isCursorLocked;
            renderer.setIsCursorLocked(isCursorLocked);
        });
    }

    // ========================== gui ==========================

    void renderGuiSection(const float deltaTime) {
        static float fps = 1 / deltaTime;

        constexpr float smoothing = 0.95f;
        fps = fps * smoothing + (1 / deltaTime) * (1.0f - smoothing);

        constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

        if (ImGui::CollapsingHeader("Engine ", sectionFlags)) {
            ImGui::Text("FPS: %.2f", fps);
        }
    }
};

int main() {
    glfwInit();

    Engine engine;
    engine.run();

    glfwTerminate();

    return EXIT_SUCCESS;
}
