#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/ubo.glsl"
#include "utils/pbr.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants {
    uint material_id;
} constants;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (set = 0, binding = 1) uniform sampler2D ssaoSampler;

#define MATERIAL_TEX_ARRAY_SIZE 32

layout (set = 1, binding = 0) uniform sampler2D baseColorSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 1) uniform sampler2D normalSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 2) uniform sampler2D ormSamplers[MATERIAL_TEX_ARRAY_SIZE];

layout (set = 2, binding = 0) uniform samplerCube irradianceMapSampler;
layout (set = 2, binding = 1) uniform samplerCube prefilterMapSampler;
layout (set = 2, binding = 2) uniform sampler2D brdfLutSampler;

float getBlurredSsao() {
    vec2 texCoord = gl_FragCoord.xy / vec2(ubo.window.width, ubo.window.height);

    vec2 texelSize = vec2(1.0) / vec2(textureSize(ssaoSampler, 0));
    float result = 0.0;

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            result += texture(ssaoSampler, texCoord + offset).r;
        }
    }

    return result / (4.0 * 4.0);
}

void main() {
    vec3 base_color = vec3(texture(baseColorSamplers[constants.material_id], fragTexCoord));

    vec3 normal = texture(normalSamplers[constants.material_id], fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

//    outColor = vec4(TBN * vec3(0.5, 0, 0) + vec3(0.5), 1.0);
//    return;

    float ao = ubo.misc.use_ssao == 1u
        ? getBlurredSsao()
        : texture(ormSamplers[constants.material_id], fragTexCoord).r;
    float roughness = texture(ormSamplers[constants.material_id], fragTexCoord).g;
    float metallic = texture(ormSamplers[constants.material_id], fragTexCoord).b;
//
//    ao = 1;
//    metallic = 0;
//    roughness = 1;

    // light related values
    vec3 light_dir = normalize(ubo.misc.light_direction);
    vec3 light_color = ubo.misc.light_color;
    vec3 radiance = light_color * ubo.misc.light_intensity;

    // utility vectors
    vec3 view = normalize(ubo.misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    // utility dot products
    float n_dot_v = max(dot(normal, view), 0.0);
    float h_dot_v = max(dot(halfway, view), 0.0);
    float n_dot_l = max(dot(normal, light_dir), 0.0);

    // BRDF intermediate values
    vec3 f0 = mix(vec3(0.04), base_color, metallic);
    vec3 fresnel = fresnel_schlick(max(dot(halfway, view), 0.0), f0);
    float ndf = distribution_ggx(normal, halfway, roughness);
    float geom = geometry_smith(normal, view, light_dir, roughness);

    // actual BRDF result
    vec3 num = ndf * geom * fresnel;
    float denom = 4.0 * max(dot(normal, view), 0.0) * max(dot(normal, light_dir), 0.0) + 0.0001;
    vec3 specular = num / denom;

    // k-terms specifying reflection vs refraction contributions
    vec3 k_specular = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 k_diffuse = (vec3(1.0) - k_specular) * (1.0 - metallic);

    vec3 out_radiance = (k_diffuse * base_color / PI + specular) * radiance * n_dot_l;

    vec3 ambient;

    if (ubo.misc.use_ibl == 1u) {
        vec3 irradiance = texture(irradianceMapSampler, normal).rgb;
        vec3 diffuse = irradiance * base_color;

        const float MAX_REFLECTION_LOD = 4.0;
        vec3 reflection = reflect(-view, normal);
        vec3 prefiltered_color = textureLod(prefilterMapSampler, reflection, roughness * MAX_REFLECTION_LOD).rgb;

        vec2 env_brdf = texture(brdfLutSampler, vec2(n_dot_v, roughness)).rg;
        specular = prefiltered_color * (k_specular * env_brdf.x + env_brdf.y);

        // no need to multiply `specular` by `k_specular` as it's done implicitly by including fresnel
        ambient = (k_diffuse * diffuse + specular) * ao;

    } else {
        ambient = vec3(0.03) * base_color * ao;
    }

    vec3 color = ambient + out_radiance;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);
}
