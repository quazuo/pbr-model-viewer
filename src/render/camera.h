#pragma once

#include <memory>

#include "libs.h"
#include "src/utils/input-manager.h"

class Rotator {
    glm::vec2 rot = {0, 0};

public:
    [[nodiscard]]
    glm::vec2 operator*() const { return rot; }

    Rotator& operator=(glm::vec2 other);

    Rotator& operator+=(glm::vec2 other);

    Rotator& operator-=(glm::vec2 other);

    struct ViewVectors {
        glm::vec3 front, right, up;
    };

    [[nodiscard]]
    ViewVectors getViewVectors() const;
};

class Camera {
    struct GLFWwindow *window = nullptr;

    float aspectRatio = 4.0f / 3.0f;
    float fieldOfView = 80.0f;
    float zNear = 0.01f;
    float zFar = 500.0f;

    glm::vec3 pos = {0, 0, -1.5};
    Rotator rotator;
    glm::vec3 front{}, right{}, up{};

    bool isLocked = true;
    float lockedRadius = 1.5f;
    Rotator lockedRotator;

    float rotationSpeed = 2.5f;
    float movementSpeed = 1.0f;

    std::unique_ptr<InputManager> inputManager;

public:
    explicit Camera(GLFWwindow *w);

    void tick(float deltaTime);

    [[nodiscard]]
    glm::vec3 getPos() const { return pos; }

    [[nodiscard]]
    glm::mat4 getViewMatrix() const;

    [[nodiscard]]
    glm::mat4 getStaticViewMatrix() const;

    [[nodiscard]]
    glm::mat4 getProjectionMatrix() const;

    void renderGuiSection();

private:
    static void scrollCallback(GLFWwindow *window, double dx, double dy);

    /**
     * Binds keys used to rotate the camera.
     */
    void bindMouseDragCallback();

    /**
     * Binds keys used to rotate the camera in freecam mode.
     */
    void bindFreecamRotationKeys();

    /**
     * Binds keys used to move the camera in freecam mode.
     */
    void bindFreecamMovementKeys();

    void tickMouseMovement(float deltaTime);

    void tickLockedMode();

    void updateAspectRatio();

    void updateVecs();

    void centerCursor() const;
};
