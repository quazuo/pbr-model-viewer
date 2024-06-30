#version 450

#include "pbr.glsl"

layout (location = 0) in vec2 texCoords;

layout (location = 0) out vec4 outColor;

vec2 integrate_brdf(float n_dot_v, float roughness) {
    vec3 view = vec3(
        sqrt(1.0 - n_dot_v * n_dot_v),
        0.0,
        n_dot_v
    );

    float A = 0.0;
    float B = 0.0;

    vec3 normal = vec3(0, 0, 1);

    const uint SAMPLE_COUNT = 1024;
    
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 x_i = hammersley(i, SAMPLE_COUNT);
        vec3 halfway = importance_sample_ggx(x_i, normal, roughness);
        vec3 light = normalize(2.0 * dot(view, halfway) * halfway - view);

        float n_dot_l = max(light.z, 0.0);
        float n_dot_h = max(halfway.z, 0.0);
        float v_dot_h = max(dot(view, halfway), 0.0);

        if (n_dot_l > 0.0) {
            float geom = geometry_smith_ibl(normal, view, light, roughness);
            float g_vis = (geom * v_dot_h) / (n_dot_h * n_dot_v + 0.001);
            float fresnel = pow(1.0 - v_dot_h, 5.0);

            A += (1.0 - fresnel) * g_vis;
            B += fresnel * g_vis;
        }
    }

    return vec2(A, B) / SAMPLE_COUNT;
}

void main() {
    vec2 integrated_brdf = integrate_brdf(texCoords.x, texCoords.y);
    outColor = vec4(integrated_brdf, 0, 1);
}
