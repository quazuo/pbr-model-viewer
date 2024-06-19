#include <iostream>

#define GLFW_INCLUDE_VULKAN
#include <ranges>

#include "GLFW/glfw3.h"

#include "render/renderer.h"
#include "utils/key-manager.h"
#include "render/gui/gui.h"

enum class FileType {
    MESH_OBJ,
    ALBEDO_PNG,
    NORMAL_PNG,
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

    void renderTexLoadButton(const std::string &label, const FileType fileType,
                             const std::vector<std::string> &typeFilters) {
        if (ImGui::Button(label.c_str(), ImVec2(180, 0))) {
            currentTypeBeingChosen = fileType;
            fileBrowser.SetTypeFilters(typeFilters);
            fileBrowser.Open();
        }

        if (chosenPaths.contains(fileType)) {
            ImGui::SameLine();
            ImGui::Text(chosenPaths.at(fileType).filename().string().c_str());
        }
    }

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
                renderTexLoadButton("Choose mesh...", FileType::MESH_OBJ, {".obj"});
                renderTexLoadButton("Choose albedo texture...", FileType::ALBEDO_PNG, {".png"});
                renderTexLoadButton("Choose normal map...", FileType::NORMAL_PNG, {".png"});
                renderTexLoadButton("Choose ORM map...", FileType::ORM_PNG, {".png"});

                ImGui::Separator();

                constexpr std::array requiredFileTypes = {
                    FileType::MESH_OBJ,
                    FileType::ALBEDO_PNG,
                    FileType::NORMAL_PNG,
                    FileType::ORM_PNG,
                };

                const bool canSubmit = std::ranges::all_of(requiredFileTypes, [&](const auto &t) {
                    return chosenPaths.contains(t);
                });

                if (!canSubmit) {
                    ImGui::BeginDisabled();
                }

                if (ImGui::Button("OK", ImVec2(120, 0))) {
                    const auto meshPath = chosenPaths.at(FileType::MESH_OBJ);
                    const auto albedoPath = chosenPaths.at(FileType::ALBEDO_PNG);
                    const auto normalPath = chosenPaths.at(FileType::ALBEDO_PNG);
                    const auto ormPath = chosenPaths.at(FileType::ORM_PNG);

                    renderer.loadModel(meshPath, albedoPath, normalPath, ormPath);
                    ImGui::CloseCurrentPopup();
                }

                if (!canSubmit) {
                    ImGui::EndDisabled();
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
