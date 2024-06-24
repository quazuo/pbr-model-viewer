#include "glfw-statics.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

void initGlfwUserPointer(GLFWwindow* window) {
    auto* userDataPtr = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!userDataPtr) {
        glfwSetWindowUserPointer(window, new GlfwStaticUserData);
    }
}
