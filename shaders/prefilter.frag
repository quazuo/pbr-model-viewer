#version 450

#define PI 3.1415926535897932384626433832795

layout (location = 0) in vec3 localPosition;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 view;
    float roughness;
} constants;

layout (binding = 1) uniform samplerCube envmapSampler;

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n_samples) {
    return vec2(float(i) / float(n_samples), radical_inverse_vdc(i));
}

vec3 importance_sample_ggx(vec2 x_i, vec3 normal, float roughness) {
    float roughness_sq = roughness * roughness;

    float phi = 2.0 * PI * x_i.x;
    float cosTheta = sqrt((1.0 - x_i.y) / (1.0 + (roughness_sq * roughness_sq - 1.0) * x_i.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    vec3 halfway;
    halfway.x = cos(phi) * sinTheta;
    halfway.y = sin(phi) * sinTheta;
    halfway.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    vec3 sampleVec = tangent * halfway.x + bitangent * halfway.y + normal * halfway.z;
    return normalize(sampleVec);
}

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
