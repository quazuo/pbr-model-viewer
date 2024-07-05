#version 450

#include "ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in mat4 inInstanceTransform;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out mat3 TBN;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    const mat4 model = ubo.matrices.model * inInstanceTransform;
    const mat4 mvp = ubo.matrices.proj * ubo.matrices.view * model;

    gl_Position = mvp * vec4(inPosition, 1.0);

    fragTexCoord = inTexCoord;

    vec3 T = normalize(vec3(model * vec4(inTangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(inNormal, 0.0)));
    T = normalize(T - dot(T, N) * N); // gramm-schmidt
    vec3 B = -cross(N, T);
    TBN = mat3(T, B, N);
}
