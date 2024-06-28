#version 450

#include "pbr.glsl"

layout (location = 0) in vec3 localPosition;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 view;
    float roughness;
} constants;

layout (binding = 1) uniform samplerCube envmapSampler;

void main() {
    vec3 normal = normalize(localPosition);
    vec3 reflection = normal;
    vec3 view = reflection;

    const uint SAMPLE_COUNT = 4096;
    float total_weight = 0;
    vec3 prefiltered_color = vec3(0);

    // monte-carlo integrate
    for (uint i = 0u; i < SAMPLE_COUNT; i++) {
        vec2 x_i = hammersley(i, SAMPLE_COUNT);
        vec3 halfway = importance_sample_ggx(x_i, normal, constants.roughness);
        vec3 light = normalize(2.0 * dot(view, halfway) * halfway - view);

        float n_dot_l = max(dot(normal, light), 0);
        if (n_dot_l > 0) {
            prefiltered_color += texture(envmapSampler, light).rgb * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    prefiltered_color /= total_weight;

    outColor = vec4(prefiltered_color, 1);
}
