#version 450

#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in mat4 inInstanceTransform;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 fragPos;
layout (location = 2) out vec3 normal;

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    const mat4 model = ubo.matrices.model * inInstanceTransform;

    vec4 view_pos = ubo.matrices.view * model * vec4(inPosition, 1.0);
    fragPos = view_pos.xyz;
    gl_Position = ubo.matrices.proj * view_pos;

    fragTexCoord = inTexCoord;

    mat3 normal_matrix = transpose(inverse(mat3(ubo.matrices.view * model)));
    normal = normal_matrix * inNormal;
}
