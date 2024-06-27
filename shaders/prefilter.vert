#version 450

#include "ubo.glsl"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 localPosition;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 view;
    float roughness;
} constants;

void main() {
    localPosition = inPosition;

    gl_Position = ubo.matrices.cubemap_capture_proj * constants.view * vec4(inPosition, 1.0);
}
