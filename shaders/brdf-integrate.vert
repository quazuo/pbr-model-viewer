#version 450

#include "ubo.glsl"

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoords;

layout(location = 0) out vec2 texCoords;

void main() {
    texCoords = inTexCoords;

    gl_Position = vec4(inPosition, 0, 1);
}
