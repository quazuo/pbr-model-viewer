#include "camera.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "gui/gui.h"

Camera::Camera(GLFWwindow *w) : window(w), keyManager(std::make_unique<KeyManager>(w)) {
    bindRotationKeys();
    bindMovementKeys();
}

void Camera::tick(const float deltaTime) {
    if (isInAutoMode) {
        // in auto mode we ignore key and mouse events
        tickAutoMode();
    } else {
        keyManager->tick(deltaTime);

        if (isCursorLocked) {
            tickMouseMovement(deltaTime);
        }
    }

    updateAspectRatio();
    updateVecs();
}

void Camera::bindRotationKeys() {
    keyManager->bindCallback(GLFW_KEY_UP, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        updateRotation(0.0f, deltaTime * rotationSpeed);
    });

    keyManager->bindCallback(GLFW_KEY_DOWN, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        updateRotation(0.0f, -deltaTime * rotationSpeed);
    });

    keyManager->bindCallback(GLFW_KEY_RIGHT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        rot.x -= deltaTime * rotationSpeed;
    });

    keyManager->bindCallback(GLFW_KEY_LEFT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        rot.x += deltaTime * rotationSpeed;
    });
}

void Camera::bindMovementKeys() {
    keyManager->bindCallback(GLFW_KEY_W, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos += front * deltaTime * movementSpeed; // Move forward
    });

    keyManager->bindCallback(GLFW_KEY_S, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos -= front * deltaTime * movementSpeed; // Move backward
    });

    keyManager->bindCallback(GLFW_KEY_D, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos += right * deltaTime * movementSpeed; // Strafe right
    });

    keyManager->bindCallback(GLFW_KEY_A, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos -= right * deltaTime * movementSpeed; // Strafe left
    });

    keyManager->bindCallback(GLFW_KEY_SPACE, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos += glm::vec3(0, 1, 0) * deltaTime * movementSpeed; // Fly upwards
    });

    keyManager->bindCallback(GLFW_KEY_LEFT_SHIFT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        pos -= glm::vec3(0, 1, 0) * deltaTime * movementSpeed; // Fly downwards
    });
}

void Camera::updateRotation(const float dx, const float dy) {
    constexpr float yAngleLimit = glm::pi<float>() / 2 - 0.1f;

    rot.x += dx;
    rot.y = std::clamp(
        rot.y + dy,
        -yAngleLimit,
        yAngleLimit
    );
}

void Camera::setIsCursorLocked(const bool b) {
    isCursorLocked = b;

    if (isCursorLocked) {
        centerCursor();
    }
}

void Camera::renderGuiSection() {
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Camera ", sectionFlags)) {
        ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("Rotation: (%.2f, %.2f)", rot.x, rot.y);

        ImGui::Separator();

        ImGui::Text("Axes:");
        if (ImGui::BeginChild("Axes", ImVec2(50, 50))) {
            drawList->AddRectFilled(
                ImGui::GetWindowPos(),
                ImGui::GetWindowPos() + ImVec2(50, 50),
                IM_COL32(0, 0, 0, 255)
            );

            const ImVec2 offset = ImGui::GetWindowPos() + ImVec2(25, 25);
            constexpr float scale = 20;
            const glm::mat4 view = getStaticViewMatrix();
            constexpr auto projectionX = glm::vec3(1, 0, 0);
            constexpr auto projectionY = glm::vec3(0, 1, 0);

            const glm::vec3 x = view * glm::vec4(1, 0, 0, 0);
            const float tx1 = scale * glm::dot(projectionX, x);
            const float tx2 = scale * glm::dot(projectionY, x);
            drawList->AddLine(offset, offset + ImVec2(tx1, -tx2), IM_COL32(255, 0, 0, 255));

            const glm::vec3 y = view * glm::vec4(0, 1, 0, 0);
            const float ty1 = scale * glm::dot(projectionX, y);
            const float ty2 = scale * glm::dot(projectionY, y);
            drawList->AddLine(offset, offset + ImVec2(ty1, -ty2), IM_COL32(0, 255, 0, 255));

            const glm::vec3 z = view * glm::vec4(0, 0, 1, 0);
            const float tz1 = scale * glm::dot(projectionX, z);
            const float tz2 = scale * glm::dot(projectionY, z);
            drawList->AddLine(offset, offset + ImVec2(tz1, -tz2), IM_COL32(0, 0, 255, 255));
        }
        ImGui::EndChild();

        ImGui::Separator();

        ImGui::SliderFloat("Field of view", &fieldOfView, 20.0f, 160.0f, "%.0f");
        ImGui::DragFloat("Rotation speed", &rotationSpeed, 1.0f, 0.0f, FLT_MAX, "%.0f");
        ImGui::DragFloat("Movement speed", &movementSpeed, 1.0f, 0.0f, FLT_MAX, "%.0f");

        ImGui::Separator();

        if (ImGui::Checkbox("Auto mode", &isInAutoMode) && isInAutoMode) {
            autoModeStopwatch.reset();
        }
        if (isInAutoMode) {
            constexpr float yAngleLimit = glm::pi<float>() / 2 - 0.1f;

            ImGui::DragFloat("Speed", &autoModeSpeed, 0.01f, 0.0f, std::numeric_limits<float>::max(), "%.2f");
            ImGui::DragFloat("Radius", &autoModeRadius, 0.01f, 0.0f, std::numeric_limits<float>::max(), "%.2f");
            ImGui::DragFloat("Y angle", &autoModeAngleY, 0.001f, -yAngleLimit, yAngleLimit, "%.3f rad");
        }
    }
}

void Camera::updateVecs() {
    front = {
        std::cos(rot.y) * std::sin(rot.x),
        std::sin(rot.y),
        std::cos(rot.y) * std::cos(rot.x)
    };

    right = glm::vec3(
        std::sin(rot.x - glm::pi<float>() / 2.0f),
        0,
        std::cos(rot.x - glm::pi<float>() / 2.0f)
    );

    up = glm::cross(right, front);
}

void Camera::tickMouseMovement(const float deltaTime) {
    (void) deltaTime;

    glm::vec<2, double> cursorPos{};
    glfwGetCursorPos(window, &cursorPos.x, &cursorPos.y);

    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    const float mouseSpeed = 0.002f * rotationSpeed;
    updateRotation(
        mouseSpeed * (static_cast<float>(windowSize.x) / 2 - static_cast<float>(cursorPos.x)),
        mouseSpeed * (static_cast<float>(windowSize.y) / 2 - static_cast<float>(cursorPos.y))
    );

    centerCursor();
}

void Camera::updateAspectRatio() {
    glm::vec<2, int> windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);
    aspectRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(pos, pos + front, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getStaticViewMatrix() const {
    return glm::lookAt(glm::vec3(0), front, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fieldOfView), aspectRatio, zNear, zFar);
}

void Camera::tickAutoMode() {
    autoModeStopwatch.tick();
    const float time = autoModeSpeed * autoModeStopwatch.getElapsed();

    pos = {
        glm::cos(autoModeAngleY) * autoModeRadius * glm::sin(time),
        glm::sin(autoModeAngleY) * autoModeRadius * -1.0f,
        glm::cos(autoModeAngleY) * autoModeRadius * glm::cos(time)
    };

    rot = {
        time - glm::pi<float>(),
        autoModeAngleY + 0.01f // this tiny term tries to alleviate the "tiny horizontal hole" issue with rendering
    };

    if (isCursorLocked) {
        // keep the cursor confined to the center as even in auto mode we don't want
        // the invisible cursor to hover on the gui
        centerCursor();
    }
}

void Camera::centerCursor() const {
    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    glfwSetCursorPos(
        window,
        static_cast<double>(windowSize.x) / 2,
        static_cast<double>(windowSize.y) / 2
    );
}
