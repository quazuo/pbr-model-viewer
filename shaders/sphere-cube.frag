#version 450

layout(location = 0) in vec3 localPosition;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D equirectangularMap;

vec2 sample_spherical_map(vec3 v) {
    const vec2 inv_atan = vec2(0.1591, 0.3183);

    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= inv_atan;
    uv += 0.5;

    return uv;
}

void main() {
    vec2 uv = sample_spherical_map(normalize(localPosition));
    vec3 color = texture(equirectangularMap, uv).rgb;

    outColor = vec4(color, 1.0);
}
