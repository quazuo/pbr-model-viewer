#version 450

#include "utils/ubo.glsl"

layout(location = 0) in vec3 texCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout(binding = 1) uniform samplerCube skyboxTexSampler;

void main() {
    vec3 color = texture(skyboxTexSampler, texCoord).rgb;

    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);

    // debug
    outColor = textureLod(skyboxTexSampler, texCoord, ubo.misc.debug_number);
    return;
}
