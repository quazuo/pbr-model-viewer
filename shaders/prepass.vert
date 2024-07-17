#version 450

#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in mat4 inInstanceTransform;

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

//    vec3 T = normalize(vec3(model * vec4(inTangent, 0.0)));
//    vec3 N = normalize(vec3(model * vec4(inNormal, 0.0)));
//    T = normalize(T - dot(T, N) * N); // gramm-schmidt
//    vec3 B = -cross(N, T);
//    TBN = mat3(T, B, N);

    mat3 normal_matrix = transpose(inverse(mat3(ubo.matrices.view * model)));
    normal = normal_matrix * inNormal;
}
