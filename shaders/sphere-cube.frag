#version 450

#include "ubo.glsl"

layout(location = 0) in vec3 localPosition;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D equirectangularMap;

vec2 SampleSphericalMap(vec3 v) {
    const vec2 invAtan = vec2(0.1591, 0.3183);

    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;

    return uv;
}

void main() {
    vec2 uv = SampleSphericalMap(normalize(localPosition));
    vec3 color = texture(equirectangularMap, uv).rgb;

    outColor = vec4(color, 1.0);
}
