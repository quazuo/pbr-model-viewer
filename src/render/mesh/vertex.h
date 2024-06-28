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

template<>
struct std::hash<Vertex> {
    size_t operator()(Vertex const &vertex) const noexcept {
        return (hash<glm::vec3>()(vertex.pos) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1) ^
               (hash<glm::vec3>()(vertex.normal) << 1);
    }
};

struct SkyboxVertex {
    glm::vec3 pos;

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

// vertices of the skybox cube.
// might change this to be generated more intelligently... but it's good enough for now
static const std::vector<SkyboxVertex> skyboxVertices = {
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{-1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}}
};

struct ScreenSpaceQuadVertex {
    glm::vec2 pos;
    glm::vec2 texCoord;

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

static const std::vector<ScreenSpaceQuadVertex> screenSpaceQuadVertices = {
    { {0, 0}, {0, 1} },
    { {0, 1}, {0, 0} },
    { {1, 0}, {1, 1} },
    { {1, 1}, {1, 0} }
};
