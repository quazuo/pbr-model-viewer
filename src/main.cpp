#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX 1
#include <iostream>
#include <random>
#include <GLFW/glfw3native.h>

#include "render/renderer.h"
#include "render/gui/gui.h"
#include "utils/input-manager.h"
#include "utils/file-type.h"

class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    std::unique_ptr<InputManager> inputManager;

    float lastTime = 0.0f;

    bool isGuiEnabled = false;
    bool showDebugQuad = false;

    ImGui::FileBrowser fileBrowser;
    std::optional<FileType> currentTypeBeingChosen;
    std::unordered_map<FileType, std::filesystem::path> chosenPaths{};
    uint32_t loadSchemeIdx = 0;

    std::string currErrorMessage;

public:
    Engine() {
        window = renderer.getWindow();

        inputManager = std::make_unique<InputManager>(window);
        bindKeyActions();
    }

    [[nodiscard]] GLFWwindow *getWindow() const { return window; }

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

        inputManager->tick(deltaTime);
        renderer.tick(deltaTime);

        if (renderer.startFrame()) {
            if (isGuiEnabled) {
                renderer.renderGui([&] {
                    renderGuiSection(deltaTime);
                    renderer.renderGuiSection();
                });
            }

            renderer.runPrepass();
            renderer.runSsaoPass();
            renderer.drawScene();

            if (showDebugQuad) {
                renderer.drawDebugQuad();
            }

            renderer.endFrame();
        }

        if (fileBrowser.HasSelected()) {
            const std::filesystem::path path = fileBrowser.GetSelected().string();

            if (*currentTypeBeingChosen == FileType::ENVMAP_HDR) {
                renderer.loadEnvironmentMap(path);
            } else {
                chosenPaths.emplace(*currentTypeBeingChosen, path);
            }

            fileBrowser.ClearSelected();
            currentTypeBeingChosen = {};
        }
    }

    void bindKeyActions() {
        inputManager->bindCallback(GLFW_KEY_GRAVE_ACCENT, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
            (void) deltaTime;
            isGuiEnabled = !isGuiEnabled;
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
#ifndef NDEBUG
            ImGui::Checkbox("show debug quad?", &showDebugQuad);
            ImGui::Separator();

            if (ImGui::Button("Reload shaders")) {
                renderer.reloadShaders();
            }
            ImGui::Separator();
#endif
            renderLoadModelPopup();
            renderModelLoadErrorPopup();
        }

        if (ImGui::CollapsingHeader("Environment ", sectionFlags)) {
            renderTexLoadButton("Choose environment map...", FileType::ENVMAP_HDR, {".hdr"});

            fileBrowser.Display();
        }
    }

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

    void renderLoadModelPopup() {
        constexpr auto comboFlags = ImGuiComboFlags_WidthFitPreview;

        if (ImGui::BeginPopupModal("Load model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Load scheme:");

            if (ImGui::BeginCombo("##scheme", fileLoadSchemes[loadSchemeIdx].name.c_str(),
                                  comboFlags)) {
                for (uint32_t i = 0; i < fileLoadSchemes.size(); i++) {
                    const bool isSelected = loadSchemeIdx == i;

                    if (ImGui::Selectable(fileLoadSchemes[i].name.c_str(), isSelected)) {
                        loadSchemeIdx = i;
                    }

                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Separator();

            for (const auto &type: fileLoadSchemes[loadSchemeIdx].requirements) {
                renderTexLoadButton(
                    getFileTypeLoadLabel(type),
                    type,
                    getFileTypeExtensions(type)
                );
            }

            ImGui::Separator();

            const bool canSubmit = std::ranges::all_of(fileLoadSchemes[loadSchemeIdx].requirements, [&](const auto &t) {
                return isFileTypeOptional(t) || chosenPaths.contains(t);
            });

            if (!canSubmit) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                loadModel();
                chosenPaths.clear();
                ImGui::CloseCurrentPopup();
            }

            if (!canSubmit) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                chosenPaths.clear();
                ImGui::CloseCurrentPopup();
            }

            fileBrowser.Display();

            ImGui::EndPopup();
        }
    }

    void loadModel() {
        const auto &reqs = fileLoadSchemes[loadSchemeIdx].requirements;

        try {
            if (reqs.contains(FileType::MODEL)) {
                renderer.loadModel(chosenPaths.at(FileType::MODEL));
            }

            if (reqs.contains(FileType::ALBEDO_PNG)) {
                renderer.loadAlbedoTexture(chosenPaths.at(FileType::ALBEDO_PNG));
            }

            if (reqs.contains(FileType::NORMAL_PNG)) {
                renderer.loadNormalMap(chosenPaths.at(FileType::NORMAL_PNG));
            }

            if (reqs.contains(FileType::ORM_PNG)) {
                renderer.loadOrmMap(chosenPaths.at(FileType::ORM_PNG));
            }

            if (reqs.contains(FileType::RMA_PNG)) {
                renderer.loadRmaMap(chosenPaths.at(FileType::RMA_PNG));
            }

            if (reqs.contains(FileType::ROUGHNESS_PNG)) {
                const auto roughnessPath = chosenPaths.at(FileType::ROUGHNESS_PNG);
                const auto aoPath = chosenPaths.contains(FileType::AO_PNG)
                                        ? chosenPaths.at(FileType::AO_PNG)
                                        : "";
                const auto metallicPath = chosenPaths.contains(FileType::METALLIC_PNG)
                                              ? chosenPaths.at(FileType::METALLIC_PNG)
                                              : "";

                renderer.loadOrmMap(aoPath, roughnessPath, metallicPath);
            }
        } catch (std::exception &e) {
            ImGui::OpenPopup("Model load error");
            currErrorMessage = e.what();
        }
    }

    void renderModelLoadErrorPopup() const {
        if (ImGui::BeginPopupModal("Model load error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("An error occurred while loading the model:");
            ImGui::Text(currErrorMessage.c_str());

            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }
};

static void showErrorBox(const std::string &message) {
    MessageBox(
        nullptr,
        static_cast<LPCSTR>(message.c_str()),
        static_cast<LPCSTR>("Error"),
        MB_OK
    );
}

void generateSsaoKernelSamples() {
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator;
    std::vector<glm::vec3> ssaoKernel;
    for (int i = 0; i < 64; ++i) {
        glm::vec3 sample(
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator)
        );
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);

        float scale = (float)i / 64.0;
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssaoKernel.push_back(sample);
    }

    for (auto &v : ssaoKernel) {
        std::cout << "vec3(" << v.x << ", " << v.y << ", " << v.z << "),\n";
    }
}

int main() {
    if (!glfwInit()) {
        showErrorBox("Fatal error: GLFW initialization failed.");
        return EXIT_FAILURE;
    }

#ifdef NDEBUG
    try {
        Engine engine;
        engine.run();
    } catch (std::exception &e) {
        showErrorBox(std::string("Fatal error: ") + e.what());
        glfwTerminate();
        return EXIT_FAILURE;
    }
#else
    Engine engine;
    engine.run();
#endif

    glfwTerminate();

    return EXIT_SUCCESS;
}
