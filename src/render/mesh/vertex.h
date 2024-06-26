#pragma once

#include "src/render/libs.h"

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;
    glm::vec3 normal;
    glm::vec3 tangent;

    bool operator==(const Vertex &other) const {
        return pos == other.pos
               && texCoord == other.texCoord
               && tangent == other.tangent;
    }

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

struct SkyboxVertex {
    glm::vec3 pos;

    bool operator==(const Vertex &other) const {
        return pos == other.pos;
    }

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

template<>
struct std::hash<Vertex> {
    size_t operator()(Vertex const &vertex) const noexcept {
        return (hash<glm::vec3>()(vertex.pos) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1) ^
               (hash<glm::vec3>()(vertex.normal) << 1);
    }
};
