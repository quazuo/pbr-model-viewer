#version 450

#include "utils/ubo.glsl"

layout (location = 0) in vec2 texCoord;
layout (location = 1) in vec3 fragPos;
layout (location = 2) in vec3 normal;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outPos;

layout (binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (binding = 1) uniform sampler2D normalSampler;

void main() {
    outNormal = normalize(normal);

    outPos = fragPos;
}
