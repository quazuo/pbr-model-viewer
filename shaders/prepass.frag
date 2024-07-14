#version 450

#include "utils/ubo.glsl"

layout (location = 0) in vec2 fragTexCoord;
layout (location = 1) in mat3 TBN;

layout (location = 0) out vec4 outNormal;

layout (binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (binding = 1) uniform sampler2D normalSampler;

void main() {
    vec3 normal = texture(normalSampler, fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);
    normal = normal * 0.5 + 0.5;
    outNormal = vec4(normal, 1);
}
