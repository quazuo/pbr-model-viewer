#version 450

#extension GL_EXT_multiview : enable

#include "utils/ubo.glsl"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 localPosition;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 view;
} constants;

void main() {
    localPosition = inPosition;

    const mat4 view = ubo.matrices.cubemap_capture_views[gl_ViewIndex];
    gl_Position = ubo.matrices.cubemap_capture_proj * view * vec4(inPosition, 1.0);
}
