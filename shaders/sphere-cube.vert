#version 450

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 localPosition;

layout(push_constant) uniform PushConstants {
    mat4 view;
    mat4 proj;
} constants;

void main() {
    localPosition = inPosition;

    gl_Position = constants.proj * constants.view * vec4(inPosition, 1.0);
}
