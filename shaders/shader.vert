#version 450

#include "ubo.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 normal;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    const mat4 vp = ubo.matrices.proj * ubo.matrices.view;

    gl_Position = vp * vec4(inPosition, 1.0);

    worldPosition = inPosition;
    fragTexCoord = inTexCoord;
    normal = normalize(inNormal);
}
