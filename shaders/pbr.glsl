#define PI 3.1415926535897932384626433832795

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

float geometry_schlick_ggx_ibl(float n_dot_v, float roughness) {
    float k = (roughness * roughness) / 2.0;

    float num = n_dot_v;
    float denom = n_dot_v * (1.0 - k) + k;

    return num / denom;
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

float geometry_smith(vec3 normal, vec3 view, vec3 light, float roughness) {
    float NdotV = max(dot(normal, view), 0.0);
    float NdotL = max(dot(normal, light), 0.0);
    float ggx1 = geometry_schlick_ggx(NdotV, roughness);
    float ggx2 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}

float geometry_smith_ibl(vec3 normal, vec3 view, vec3 light, float roughness) {
    float NdotV = max(dot(normal, view), 0.0);
    float NdotL = max(dot(normal, light), 0.0);
    float ggx1 = geometry_schlick_ggx_ibl(NdotV, roughness);
    float ggx2 = geometry_schlick_ggx_ibl(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

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
