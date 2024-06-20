#pragma once

#include "src/render/libs.h"

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;
    glm::vec3 normal;

    bool operator==(const Vertex &other) const {
        return pos == other.pos && texCoord == other.texCoord && normal == other.normal;
    }

    static std::array<vk::VertexInputBindingDescription, 2> getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 7> getAttributeDescriptions();
};

struct SkyboxVertex {
    glm::vec3 pos;

    bool operator==(const Vertex &other) const {
        return pos == other.pos;
    }

    static vk::VertexInputBindingDescription getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions();
};

template<>
struct std::hash<Vertex> {
    size_t operator()(Vertex const &vertex) const noexcept {
        return (hash<glm::vec3>()(vertex.pos) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1) ^
               (hash<glm::vec3>()(vertex.normal) << 1);
    }
};
