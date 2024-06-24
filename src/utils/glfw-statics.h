#pragma once

struct GlfwStaticUserData {
    class VulkanRenderer* renderer;
    class Camera* camera;
};

void initGlfwUserPointer(struct GLFWwindow* window);
