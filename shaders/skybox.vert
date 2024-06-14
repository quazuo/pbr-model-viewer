#version 450

#include "ubo.glsl"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 texCoord;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    texCoord = inPosition;

    const vec4 pos = ubo.matrices.proj * ubo.matrices.static_view * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;
}
