#include "camera.h"

#define GLFW_INCLUDE_VULKAN
#include <iostream>
#include <GLFW/glfw3.h>

#include "gui/gui.h"
#include "src/utils/glfw-statics.h"

Rotator &Rotator::operator=(const glm::vec2 other) {
    rot = other;
    return *this;
}

Rotator &Rotator::operator+=(const glm::vec2 other) {
    static constexpr float yAngleLimit = glm::pi<float>() / 2 - 0.1f;

    rot.x += other.x;
    rot.y = std::clamp(
        rot.y + other.y,
        -yAngleLimit,
        yAngleLimit
    );

    return *this;
}

Rotator &Rotator::operator-=(const glm::vec2 other) {
    *this += -other;
    return *this;
}

Rotator::ViewVectors Rotator::getViewVectors() const {
    const glm::vec3 front = {
        std::cos(rot.y) * std::sin(rot.x),
        std::sin(rot.y),
        std::cos(rot.y) * std::cos(rot.x)
    };

    const glm::vec3 right = {
        std::sin(rot.x - glm::pi<float>() / 2.0f),
        0,
        std::cos(rot.x - glm::pi<float>() / 2.0f)
    };

    return {
        .front = front,
        .right = right,
        .up = glm::cross(right, front)
    };
}

Camera::Camera(GLFWwindow *w) : window(w), inputManager(make_unique<InputManager>(w)) {
    bindCameraLockKey();
    bindFreecamMovementKeys();
    bindFreecamRotationKeys();
    bindMouseDragCallback();

    initGlfwUserPointer(window);
    auto *userData = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!userData) throw std::runtime_error("unexpected null window user pointer");
    userData->camera = this;

    glfwSetScrollCallback(window, &scrollCallback);
}

void Camera::tick(const float deltaTime) {
    if (
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)
        && !ImGui::IsAnyItemActive()
        && !ImGui::IsAnyItemFocused()
    ) {
        inputManager->tick(deltaTime);
    }

    if (isLockedCam) {
        tickLockedMode();
    } else if (isLockedCursor) {
        tickMouseMovement(deltaTime);
    }

    updateAspectRatio();
    updateVecs();
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

void Camera::renderGuiSection() {
    ImDrawList *drawList = ImGui::GetWindowDrawList();

    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Camera ", sectionFlags)) {
        ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("Rotation: (%.2f, %.2f)", (*rotator).x, (*rotator).y);

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

        if (ImGui::RadioButton("Free camera", !isLockedCam)) {
            isLockedCam = false;
        }

        ImGui::SameLine();

        if (ImGui::RadioButton("Locked camera", isLockedCam)) {
            isLockedCam = true;

            if (isLockedCursor) {
                centerCursor();
            }
        }

        ImGui::Separator();

        ImGui::SliderFloat("Field of view", &fieldOfView, 20.0f, 160.0f, "%.0f");

        if (!isLockedCam) {
            ImGui::DragFloat("Rotation speed", &rotationSpeed, 0.01f, 0.0f, FLT_MAX, "%.2f");
            ImGui::DragFloat("Movement speed", &movementSpeed, 0.01f, 0.0f, FLT_MAX, "%.2f");
        }
    }
}

void Camera::scrollCallback(GLFWwindow *window, const double dx, const double dy) {
    const auto userData = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!userData) throw std::runtime_error("unexpected null window user pointer");

    if (
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)
        && !ImGui::IsAnyItemActive()
        && !ImGui::IsAnyItemFocused()
    ) {
        userData->camera->lockedRadius /= static_cast<float>(1 + dy * 0.05);
    }
}

void Camera::bindCameraLockKey() {
    inputManager->bindCallback(GLFW_KEY_F1, EActivationType::PRESS_ONCE, [&](const float deltaTime) {
        (void) deltaTime;
        if (isLockedCam) {
            return;
        }

        isLockedCursor = !isLockedCursor;

        if (isLockedCursor) {
            centerCursor();
        }
    });
}

void Camera::bindMouseDragCallback() {
    inputManager->bindMouseDragCallback(GLFW_MOUSE_BUTTON_LEFT, [&](const double dx, const double dy) {
        if (isLockedCam) {
            static constexpr float speed = 0.003;

            lockedRotator += {
                -speed * static_cast<float>(dx),
                -speed * static_cast<float>(dy)
            };
        }
    });
}

void Camera::bindFreecamRotationKeys() {
    inputManager->bindCallback(GLFW_KEY_UP, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            rotator += glm::vec2(0, deltaTime * rotationSpeed);
        }
    });

    inputManager->bindCallback(GLFW_KEY_DOWN, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            rotator -= glm::vec2(0, deltaTime * rotationSpeed);
        }
    });

    inputManager->bindCallback(GLFW_KEY_RIGHT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            rotator -= glm::vec2(deltaTime * rotationSpeed, 0);
        }
    });

    inputManager->bindCallback(GLFW_KEY_LEFT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            rotator += glm::vec2(deltaTime * rotationSpeed, 0);
        }
    });
}

void Camera::bindFreecamMovementKeys() {
    inputManager->bindCallback(GLFW_KEY_W, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos += front * deltaTime * movementSpeed; // Move forward
        }
    });

    inputManager->bindCallback(GLFW_KEY_S, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos -= front * deltaTime * movementSpeed; // Move backward
        }
    });

    inputManager->bindCallback(GLFW_KEY_D, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos += right * deltaTime * movementSpeed; // Strafe right
        }
    });

    inputManager->bindCallback(GLFW_KEY_A, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos -= right * deltaTime * movementSpeed; // Strafe left
        }
    });

    inputManager->bindCallback(GLFW_KEY_SPACE, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos += glm::vec3(0, 1, 0) * deltaTime * movementSpeed; // Fly upwards
        }
    });

    inputManager->bindCallback(GLFW_KEY_LEFT_SHIFT, EActivationType::PRESS_ANY, [&](const float deltaTime) {
        if (!isLockedCam) {
            pos -= glm::vec3(0, 1, 0) * deltaTime * movementSpeed; // Fly downwards
        }
    });
}

void Camera::tickMouseMovement(const float deltaTime) {
    (void) deltaTime;

    glm::vec<2, double> cursorPos{};
    glfwGetCursorPos(window, &cursorPos.x, &cursorPos.y);

    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    const float mouseSpeed = 0.002f * rotationSpeed;

    rotator += {
        mouseSpeed * (static_cast<float>(windowSize.x / 2) - static_cast<float>(std::floor(cursorPos.x))),
        mouseSpeed * (static_cast<float>(windowSize.y / 2) - static_cast<float>(std::floor(cursorPos.y)))
    };

    centerCursor();
}

void Camera::tickLockedMode() {
    const glm::vec2 rot = *lockedRotator;

    pos = {
        glm::cos(rot.y) * lockedRadius * glm::sin(rot.x),
        glm::sin(rot.y) * lockedRadius * -1.0f,
        glm::cos(rot.y) * lockedRadius * glm::cos(rot.x)
    };

    rotator = {
        rot.x - glm::pi<float>(),
        rot.y
    };
}

void Camera::updateVecs() {
    const Rotator::ViewVectors viewVectors = rotator.getViewVectors();

    front = viewVectors.front;
    right = viewVectors.right;
    up = viewVectors.up;
}

void Camera::updateAspectRatio() {
    glm::vec<2, int> windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    if (windowSize.y == 0) {
        aspectRatio = 1;
    } else {
        aspectRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
    }
}

void Camera::centerCursor() const {
    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    glfwSetCursorPos(
        window,
        windowSize.x / 2,
        windowSize.y / 2
    );
}
