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

layout(binding = 1) uniform sampler2D albedoSampler;
layout(binding = 2) uniform sampler2D normalSampler;
layout(binding = 3) uniform sampler2D ormSampler;

float get_ao() {
    return texture(ormSampler, fragTexCoord).x;
}

float get_roughness() {
    return texture(ormSampler, fragTexCoord).y;
}

float get_metallic() {
    return texture(ormSampler, fragTexCoord).z;
}

void main() {
    outColor = texture(albedoSampler, fragTexCoord);

    outColor *= clamp(dot(normal, normalize(lightDirection)), 0, 1);

    outColor = pow(outColor, vec4(1 / 2.2));
}
