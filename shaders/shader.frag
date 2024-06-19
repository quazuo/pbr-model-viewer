#version 450

#include "ubo.glsl"

#define PI 3.1415926535897932384626433832795

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in vec3 normal;

layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (binding = 1) uniform sampler2D albedoSampler;
layout (binding = 2) uniform sampler2D normalSampler;
layout (binding = 3) uniform sampler2D ormSampler;

float distribution_ggx(vec3 normal, vec3 halfway, float roughness) {
    float roughness_sq = roughness * roughness;
    float n_dot_h = max(dot(normal, halfway), 0.0);

    float num = roughness_sq;
    float denom = (n_dot_h * n_dot_h * (roughness_sq - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float num = n_dot_v;
    float denom = n_dot_v * (1.0 - k) + k;

    return num / denom;
}

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness) {
    float NdotV = max(dot(normal, view), 0.0);
    float NdotL = max(dot(normal, light), 0.0);
    float ggx1 = geometry_schlick_ggx(NdotV, roughness);
    float ggx2 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 albedo = vec3(texture(albedoSampler, fragTexCoord));
    float ao = texture(ormSampler, fragTexCoord).r;
    float roughness = texture(ormSampler, fragTexCoord).g;
    float metallic = texture(ormSampler, fragTexCoord).b;

    // light related values
    vec3 light_dir = ubo.misc.light_direction;
    vec3 light_color = vec3(23.47, 21.31, 20.79);
    vec3 wi = vec3(0, 1, 0);
    float cos_theta = max(dot(normal, wi), 0.0);
    vec3 radiance = light_color * cos_theta;

    // utility vectors
    vec3 view = normalize(ubo.misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    // BRDF intermediate values
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = fresnel_schlick(max(dot(halfway, view), 0.0), f0);
    float ndf = distribution_ggx(normal, halfway, roughness);
    float geom = geometry_smith(normal, view, light_dir, roughness);

    // actual BRDF result
    vec3 num = ndf * geom * fresnel;
    float denom = 4.0 * max(dot(normal, view), 0.0) * max(dot(normal, light_dir), 0.0) + 0.0001;
    vec3 specular = num / denom;

    // k-terms specifying reflection vs refraction contributions
    vec3 k_specular = fresnel;
    vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - metallic);

    float n_dot_l = max(dot(normal, light_dir), 0.0);
    vec3 out_radiance = (k_diffuse * albedo / PI + specular) * radiance * n_dot_l;
    vec3 ambient = vec3(0.03) * albedo * ao;
    outColor = vec4(ambient + out_radiance, 1.0);

    // apply hdr
    outColor = outColor / (outColor + vec4(1.0));

    // apply gamma correction
    outColor = pow(outColor, vec4(1 / 2.2));
}
