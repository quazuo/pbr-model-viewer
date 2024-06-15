#include <iostream>

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#include "render/renderer.h"
#include "utils/key-manager.h"
#include "render/gui/gui.h"

enum class FileType {
    MESH_OBJ,
    ALBEDO_PNG,
    ORM_PNG,
    RMA_PNG,
};

class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    std::unique_ptr<KeyManager> keyManager;

    float lastTime = 0.0f;

    bool isCursorLocked = true;
    bool doShowGui = false;

    ImGui::FileBrowser fileBrowser;
    std::optional<FileType> currentTypeBeingChosen;
    std::unordered_map<FileType, std::filesystem::path> chosenPaths{};

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

        if (fileBrowser.HasSelected()) {
            const std::filesystem::path path = fileBrowser.GetSelected().string();
            chosenPaths.emplace(*currentTypeBeingChosen, path);
            fileBrowser.ClearSelected();
            currentTypeBeingChosen = {};
        }
    }

    void bindKeyActions() {
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

        if (ImGui::CollapsingHeader("Model ", sectionFlags)) {
            if (ImGui::Button("Load model...")) {
                chosenPaths.clear();
                ImGui::OpenPopup("Load model");
            }

            if (ImGui::BeginPopupModal("Load model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::Button("Choose mesh...", ImVec2(150, 0))) {
                    currentTypeBeingChosen = FileType::MESH_OBJ;
                    fileBrowser.SetTypeFilters({".obj"});
                    fileBrowser.Open();
                }

                if (chosenPaths.contains(FileType::MESH_OBJ)) {
                    ImGui::SameLine();
                    ImGui::Text(chosenPaths.at(FileType::MESH_OBJ).filename().string().c_str());
                }

                if (ImGui::Button("Choose texture...", ImVec2(150, 0))) {
                    currentTypeBeingChosen = FileType::ALBEDO_PNG;
                    fileBrowser.SetTypeFilters({".png"});
                    fileBrowser.Open();
                }

                if (chosenPaths.contains(FileType::ALBEDO_PNG)) {
                    ImGui::SameLine();
                    ImGui::Text(chosenPaths.at(FileType::ALBEDO_PNG).filename().string().c_str());
                }

                ImGui::Separator();

                if (ImGui::Button("OK", ImVec2(120, 0))) {
                    const auto meshPath = chosenPaths.at(FileType::MESH_OBJ);
                    const auto texturePath = chosenPaths.at(FileType::ALBEDO_PNG);

                    renderer.loadModel(meshPath, texturePath);
                    ImGui::CloseCurrentPopup();
                }

                ImGui::SameLine();

                if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                    ImGui::CloseCurrentPopup();
                }

                fileBrowser.Display();

                ImGui::EndPopup();
            }
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
