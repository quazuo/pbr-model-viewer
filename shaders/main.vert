#version 450

#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in mat4 inInstanceTransform;

layout (location = 0) out vec3 worldPosition;
layout (location = 1) out vec2 fragTexCoord;
layout (location = 2) out mat3 TBN;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    const mat4 model = ubo.matrices.model * inInstanceTransform;
    const mat4 mvp = ubo.matrices.proj * ubo.matrices.view * model;

    gl_Position = mvp * vec4(inPosition, 1.0);

    worldPosition = (model * vec4(inPosition, 1.0)).xyz;
    fragTexCoord = inTexCoord;

    mat3 normal_matrix = transpose(inverse(mat3(model)));

    vec3 T = normalize(normal_matrix * inTangent);
    vec3 B = normalize(normal_matrix * inBitangent);
    vec3 N = normalize(normal_matrix * inNormal);

    TBN = mat3(T, B, N);
}
