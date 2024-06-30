#version 450

#include "ubo.glsl"
#include "pbr.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (binding = 1) uniform sampler2D albedoSampler;
layout (binding = 2) uniform sampler2D normalSampler;
layout (binding = 3) uniform sampler2D ormSampler;
layout (binding = 4) uniform samplerCube irradianceMapSampler;
layout (binding = 5) uniform samplerCube prefilterMapSampler;
layout (binding = 6) uniform sampler2D brdfLutSampler;

void main() {
    vec3 albedo = vec3(texture(albedoSampler, fragTexCoord));

    vec3 normal = texture(normalSampler, fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    float ao = texture(ormSampler, fragTexCoord).r;
    float roughness = texture(ormSampler, fragTexCoord).g;
    float metallic = texture(ormSampler, fragTexCoord).b;

    // light related values
    vec3 light_dir = normalize(ubo.misc.light_direction);
    vec3 light_color = vec3(23.47, 21.31, 20.79);
    vec3 radiance = vec3(0); // light_color; // temporarily disable the directional light

    // utility vectors
    vec3 view = normalize(ubo.misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    // utility dot products
    float n_dot_v = max(dot(normal, view), 0.0);
    float h_dot_v = max(dot(halfway, view), 0.0);
    float n_dot_l = max(dot(normal, light_dir), 0.0);

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
    vec3 k_specular = ubo.misc.use_ibl == 1u
        ? fresnel_schlick_roughness(n_dot_v, f0, roughness)
        : fresnel;
    vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - metallic);

    vec3 out_radiance = (k_diffuse * albedo / PI + specular) * radiance * n_dot_l;

    vec3 irradiance = texture(irradianceMapSampler, normal).rgb;
    vec3 diffuse = irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 reflection = reflect(-view, normal);
    vec3 prefiltered_color = textureLod(prefilterMapSampler, reflection, roughness * MAX_REFLECTION_LOD).rgb;

    vec2 env_brdf = texture(brdfLutSampler, vec2(n_dot_v, roughness)).rg;
    vec3 specular = prefiltered_color * (k_specular * env_brdf.x + env_brdf.y);

    // no need to multiply `specular` by `k_specular` as it's done implicitly by including fresnel
    vec3 ambient = (k_diffuse * diffuse + specular) * ao;

    vec3 color = ambient + out_radiance;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);
}
