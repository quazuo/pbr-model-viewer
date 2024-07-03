#include "input-manager.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

void InputManager::bindCallback(const EKey k, const EActivationType type, const EInputCallback& f) {
    callbackMap.emplace(k, std::make_pair(type, f));
    keyStateMap.emplace(k, KeyState::RELEASED);
}

void InputManager::bindMouseDragCallback(const EMouseButton button, const EMouseDragCallback &f) {
    mouseDragCallbackMap.emplace(button, f);
    mouseButtonStateMap.emplace(button, KeyState::RELEASED);
}

void InputManager::tick(const float deltaTime) {
    for (const auto &[key, callbackInfo]: callbackMap) {
        const auto &[activationType, callback] = callbackInfo;

        if (checkKey(key, activationType)) {
            callback(deltaTime);
        }
    }

    glm::dvec2 mousePos;
    glfwGetCursorPos(window, &mousePos.x, &mousePos.y);

    for (const auto &[button, callback]: mouseDragCallbackMap) {
        if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
            if (mouseButtonStateMap.at(button) == KeyState::PRESSED) {
                const glm::dvec2 mousePosDelta = mousePos - lastMousePos;
                callback(mousePosDelta.x, mousePosDelta.y);

            } else {
                mouseButtonStateMap[button] = KeyState::PRESSED;
            }
        } else {
            mouseButtonStateMap[button] = KeyState::RELEASED;
        }
    }

    lastMousePos = mousePos;
}

static bool isPressed(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_PRESS || glfwGetMouseButton(window, key) == GLFW_PRESS;
}

static bool isReleased(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_RELEASE || glfwGetMouseButton(window, key) == GLFW_RELEASE;
}

bool InputManager::checkKey(const EKey key, const EActivationType type) {
    if (type == EActivationType::PRESS_ANY) {
        return isPressed(window, key);
    }

    if (type == EActivationType::RELEASE_ONCE) {
        return isReleased(window, key);
    }

    if (type == EActivationType::PRESS_ONCE) {
        if (isPressed(window, key)) {
            const bool isOk = keyStateMap[key] == KeyState::RELEASED;
            keyStateMap[key] = KeyState::PRESSED;
            return isOk;
        }

        keyStateMap[key] = KeyState::RELEASED;
    }

    return false;
}
