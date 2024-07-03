#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX 1
#include <GLFW/glfw3native.h>

#include "render/renderer.h"
#include "utils/input-manager.h"
#include "render/gui/gui.h"

enum class FileType {
    MODEL,
    ALBEDO_PNG,
    NORMAL_PNG,
    ORM_PNG,
    RMA_PNG,
    AO_PNG,
    ROUGHNESS_PNG,
    METALLIC_PNG,
    ENVMAP_HDR,
};

[[nodiscard]] static std::vector<std::string> getFileTypeExtensions(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return { ".obj", ".fbx" };
        case FileType::ALBEDO_PNG:
        case FileType::NORMAL_PNG:
        case FileType::ORM_PNG:
        case FileType::RMA_PNG:
        case FileType::AO_PNG:
        case FileType::ROUGHNESS_PNG:
        case FileType::METALLIC_PNG:
            return { ".png" };
        case FileType::ENVMAP_HDR:
            return { ".hdr" };
        default:
            throw std::runtime_error("unexpected filetype in getFileTypeExtensions");
    }
}

[[nodiscard]] static bool isFileTypeOptional(const FileType type) {
    switch (type) {
        case FileType::AO_PNG:
        case FileType::METALLIC_PNG:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static std::string getFileTypeLoadLabel(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return "Load model...";
        case FileType::ALBEDO_PNG:
            return "Load color texture...";
        case FileType::NORMAL_PNG:
            return "Load normal map...";
        case FileType::ORM_PNG:
            return "Load ORM map...";
        case FileType::RMA_PNG:
            return "Load RMA map...";
        case FileType::AO_PNG:
            return "Load AO map...";
        case FileType::ROUGHNESS_PNG:
            return "Load roughness map...";
        case FileType::METALLIC_PNG:
            return "Load metallic map...";
        case FileType::ENVMAP_HDR:
            return "Load environment map...";
        default:
            throw std::runtime_error("unexpected filetype in getFileTypeLoadLabel");
    }
}

struct FileLoadScheme {
    std::string name;
    std::set<FileType> requirements;
};

static const std::vector<FileLoadScheme> fileLoadSchemes{
    {
        "Albedo + Normal + ORM",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::ORM_PNG,
        }
    },
    {
        "Albedo + Normal + RMA",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::RMA_PNG,
        }
    },
    {
        "Albedo + Normal + AO + Roughness + Metallic",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::AO_PNG,
            FileType::ROUGHNESS_PNG,
            FileType::METALLIC_PNG
        }
    },
};

class Engine {
    GLFWwindow *window = nullptr;
    VulkanRenderer renderer;
    std::unique_ptr<InputManager> inputManager;

    float lastTime = 0.0f;

    bool isGuiEnabled = false;

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

            renderer.drawScene();

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

            for (const auto& type: fileLoadSchemes[loadSchemeIdx].requirements) {
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
        const auto& reqs = fileLoadSchemes[loadSchemeIdx].requirements;

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

            renderer.rebuildDescriptors();

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

int main() {
    if (!glfwInit()) {
        showErrorBox("Fatal error: GLFW initialization failed.");
        return EXIT_FAILURE;
    }

    try {
        Engine engine;
        engine.run();
    } catch (std::exception &e) {
        showErrorBox(std::string("Fatal error: ") + e.what());
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwTerminate();

    return EXIT_SUCCESS;
}
