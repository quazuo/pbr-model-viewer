#include "key-manager.h"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

void KeyManager::bindCallback(const EKey k, const EActivationType type, const EKeyCallback& f) {
    callbackMap.emplace(k, std::make_pair(type, f));
    keyStateMap.emplace(k, KeyState::RELEASED);
}

void KeyManager::tick(const float deltaTime) {
    for (const auto &[key, callbackInfo]: callbackMap) {
        const auto &[activationType, callback] = callbackInfo;

        if (checkKey(key, activationType)) {
            callback(deltaTime);
        }
    }
}

static bool isPressed(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_PRESS || glfwGetMouseButton(window, key) == GLFW_PRESS;
}

static bool isReleased(GLFWwindow* window, const EKey key) {
    return glfwGetKey(window, key) == GLFW_RELEASE || glfwGetMouseButton(window, key) == GLFW_RELEASE;
}

bool KeyManager::checkKey(const EKey key, const EActivationType type) {
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
