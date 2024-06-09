#include <iostream>

#include "render/renderer.h"
#include "utils/key-manager.h"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

class Engine {
    AutomatonConfig automatonConfig;
    AutomatonConfig uncommitedConfig;
    bool isConfigDirty = true;

    GLFWwindow *window = nullptr;
    AutomatonRenderer renderer;
    std::unique_ptr<KeyManager> keyManager;

    float lastTime = 0.0f;

    bool isCursorLocked = true;
    bool doShowGui = false;

    bool doTransitions = true;
    float automatonTransitionTime = 0.1f;
    Stopwatch automatonTransitionStopwatch{ automatonTransitionTime };

    unique_ptr<OctreeGen::Preset> genPreset = make_unique<OctreeGen::UniformPreset>(0.5f);

    bool shouldRerunAutomaton = false;

public:
    Engine() : renderer(automatonConfig) {
        window = renderer.getWindow();

        renderer.setIsCursorLocked(isCursorLocked);
        renderer.setDoShowGui(doShowGui);

        initAutomaton();

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
        automatonTransitionStopwatch.tick();

        if (shouldRerunAutomaton) {
            rerunAutomaton();
            shouldRerunAutomaton = false;
        }

        if (isConfigDirty) {
            renderer.updateAutomatonConfig(automatonConfig);
            isConfigDirty = false;
        }

        const bool shouldDoTransition = doTransitions && automatonTransitionStopwatch.isFinished();
        if (shouldDoTransition) {
            renderer.runCompute();
            automatonTransitionStopwatch.reset();
        }

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

    void initAutomaton() {
        renderer.waitIdle();

        const OctreeGen::OctreeBuf initValues = genPreset->generate(automatonConfig.gridDepth);
        renderer.fillSsbos(initValues);
    }

    void rerunAutomaton() {
        const bool shouldRebuildSsbos = automatonConfig.gridDepth != uncommitedConfig.gridDepth;
        automatonConfig = uncommitedConfig;
        renderer.updateAutomatonConfig(automatonConfig);

        automatonTransitionStopwatch.reset();

        if (shouldRebuildSsbos) {
            renderer.rebuildSsbos();
        }

        initAutomaton();
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

        if (ImGui::CollapsingHeader("Automaton ", sectionFlags)) {
            ImGui::Checkbox("Do transitions?", &doTransitions);

            const bool wasDragged = ImGui::DragFloat("Transition time", &automatonTransitionTime, 0.01f, 0.0f,
                                                     std::numeric_limits<float>::max(), "%.2fs");
            if (wasDragged) {
                automatonTransitionStopwatch = Stopwatch(automatonTransitionTime);
            }

            ImGui::Separator();

            ImGui::Text("Grid depth: %d ", uncommitedConfig.gridDepth);
            ImGui::SameLine();
            if (ImGui::ArrowButton("griddepth_left", ImGuiDir_Left)) {
                uncommitedConfig.gridDepth = std::max(0u, uncommitedConfig.gridDepth - 1);
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("griddepth_right", ImGuiDir_Right)) {
                uncommitedConfig.gridDepth = std::min(10u, uncommitedConfig.gridDepth + 1);
            }

            ImGui::Separator();

            static int selectedPresetIdx = 0;
            const int customPresetIdx = automatonPresets.size() - 1;
            constexpr auto comboFlags = ImGuiComboFlags_WidthFitPreview;

            ImGui::Text("Automaton preset:");
            if (ImGui::BeginCombo("##automaton_preset", automatonPresets[selectedPresetIdx].first.c_str(),
                                  comboFlags)) {
                for (int i = 0; i < automatonPresets.size(); i++) {
                    const bool isSelected = selectedPresetIdx == i;

                    if (ImGui::Selectable(automatonPresets[i].first.c_str(), isSelected)) {
                        if (i != customPresetIdx) {
                            uncommitedConfig.preset = automatonPresets[i].second;
                        }

                        selectedPresetIdx = i;
                    }

                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Text(uncommitedConfig.preset.toString().c_str());

            if (selectedPresetIdx == customPresetIdx) {
                ImGui::Separator();
                renderGuiCustomPresetSettings();
            }

            ImGui::Separator();

            renderGuiInitialStateSettings();

            ImGui::Separator();

            if (ImGui::Button("Run!")) {
                shouldRerunAutomaton = true;
            }
        }
    }

    void renderGuiCustomPresetSettings() {
        auto &[surviveMask, birthMask, stateCount, useMooreNeighborhood] = uncommitedConfig.preset;

        ImGui::Text("State count: %d ", stateCount);
        ImGui::SameLine();
        if (ImGui::ArrowButton("statecount_left", ImGuiDir_Left)) {
            stateCount = std::max(0u, stateCount - 1);
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("statecount_right", ImGuiDir_Right)) {
            stateCount = std::min(9u, stateCount + 1);
        }

        ImGui::Separator();

        ImGui::Text("Neighborhood type: ");
        ImGui::SameLine();
        if (ImGui::Button(useMooreNeighborhood ? "Moore" : "Von Neumann")) {
            useMooreNeighborhood ^= 1;
        }

        ImGui::Separator();

        const int maskSelectSize = useMooreNeighborhood ? 27 : 7;

        ImGui::Text("Survive conditions: ");
        for (int i = 0; i < maskSelectSize; i++) {
            const bool isSelected = (surviveMask & (1 << i)) != 0;

            ImGui::PushID(i);
            if (ImGui::Selectable(std::to_string(i).c_str(), isSelected, 0, ImVec2(20, 20))) {
                surviveMask ^= 1 << i;
            }
            ImGui::PopID();

            if (i % 8 != 7 && i != maskSelectSize - 1) ImGui::SameLine();
        }

        ImGui::Separator();

        ImGui::Text("Birth conditions: ");
        for (int i = 0; i < maskSelectSize; i++) {
            const bool isSelected = (birthMask & (1 << i)) != 0;

            ImGui::PushID(maskSelectSize + i);
            if (ImGui::Selectable(std::to_string(i).c_str(), isSelected, 0, ImVec2(20, 20))) {
                birthMask ^= 1 << i;
            }
            ImGui::PopID();

            if (i % 8 != 7 && i != maskSelectSize - 1) ImGui::SameLine();
        }
    }

    void renderGuiInitialStateSettings() {
        static int selectedPresetIdx = 0;

        const std::vector<std::pair<std::string, std::function<void(bool)> > > initStatePresets{
            {
                "Random (uniform)", [&](const bool didJustSelect) {
                    static float prob = 0.5001f;
                    if (ImGui::DragFloat("Probability", &prob, 0.001f, 0.0f, 1.0f) || didJustSelect) {
                        genPreset = make_unique<OctreeGen::UniformPreset>(prob);
                    }
                }
            },
            {
                "Centered sphere", [&](const bool didJustSelect) {
                    static float radius = 10.0f;
                    const bool wasChanged =
                            ImGui::DragFloat("Radius", &radius, 0.1f, 0.0f, std::numeric_limits<float>::max(), "%.1f");
                    if (wasChanged || didJustSelect) {
                        genPreset = make_unique<OctreeGen::SpherePreset>(radius);
                    }
                }
            },
            {
                "Centered cube", [&](const bool didJustSelect) {
                    static int sideLength = 10.0f;
                    const bool wasChanged =
                            ImGui::DragInt("Side length", &sideLength, 1, 0, std::numeric_limits<int>::max());
                    if (wasChanged || didJustSelect) {
                        genPreset = make_unique<OctreeGen::CubePreset>(static_cast<std::uint32_t>(sideLength));
                    }
                }
            }
        };

        bool didJustSelect = false;
        constexpr auto comboFlags = ImGuiComboFlags_WidthFitPreview;

        ImGui::Text("Initial state generation:");
        if (ImGui::BeginCombo("##init_state_preset", initStatePresets[selectedPresetIdx].first.c_str(), comboFlags)) {
            didJustSelect = false;

            for (int i = 0; i < initStatePresets.size(); i++) {
                const bool isSelected = selectedPresetIdx == i;

                if (ImGui::Selectable(initStatePresets[i].first.c_str(), isSelected)) {
                    selectedPresetIdx = i;
                    didJustSelect = true;
                }

                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        initStatePresets[selectedPresetIdx].second(didJustSelect);
    }
};

int main() {
    glfwInit();

    Engine engine;
    engine.run();

    glfwTerminate();

    return EXIT_SUCCESS;
}
