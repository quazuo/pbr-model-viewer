#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 lightDirection;

struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
};

struct MiscData {
    vec3 camera_pos;
};

layout(binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

void main() {
    const mat4 vp = ubo.matrices.proj * ubo.matrices.view;

    gl_Position = vp * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    normal = inNormal;
    lightDirection = ubo.misc.camera_pos - inPosition;
}
