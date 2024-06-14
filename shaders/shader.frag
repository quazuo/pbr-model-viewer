#version 450

#include "ubo.glsl"

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 lightDirection;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

void main() {
    outColor = texture(texSampler, fragTexCoord);

    outColor *= clamp(dot(normal, normalize(lightDirection)), 0, 1);

    outColor = pow(outColor, vec4(1 / 2.2));
}
