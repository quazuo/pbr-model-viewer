#version 450

#include "utils/pbr.glsl"

layout (location = 0) in vec3 localPosition;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float roughness;
} constants;

layout (binding = 1) uniform samplerCube envmapSampler;

void main() {
    vec3 normal = normalize(localPosition);
    vec3 reflection = normal;
    vec3 view = reflection;

    float resolution = 2048.0;
    float sa_texel = 4.0 * PI / (6.0 * resolution * resolution);

    const uint SAMPLE_COUNT = 2048;
    float total_weight = 0;
    vec3 prefiltered_color = vec3(0);

    // monte-carlo integrate
    for (uint i = 0u; i < SAMPLE_COUNT; i++) {
        vec2 x_i = hammersley(i, SAMPLE_COUNT);
        vec3 halfway = importance_sample_ggx(x_i, normal, constants.roughness);
        vec3 light = normalize(2.0 * dot(view, halfway) * halfway - view);

        float n_dot_h = max(dot(normal, halfway), 0.0);
        float h_dot_v = max(dot(halfway, view), 0.0);
        float distr = distribution_ggx(normal, halfway, constants.roughness);
        float pdf = distr * n_dot_h / (4.0 * h_dot_v) + 0.0001;

        float sa_sample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);
        float mip_level = constants.roughness == 0.0 ? 0.0 : 0.5 * log2(sa_sample / sa_texel);

        float n_dot_l = max(dot(normal, light), 0);
        prefiltered_color += clamp(textureLod(envmapSampler, light, mip_level).rgb, 0, 10) * n_dot_l;
        total_weight += n_dot_l;
    }

    prefiltered_color /= total_weight;

    outColor = vec4(prefiltered_color, 1.0);
}
