#pragma once

#include "src/render/libs.h"

struct ModelVertex {
    glm::vec3 pos;
    glm::vec2 texCoord;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;

    // this is implemented to allow using `Vertex` as a key in an `unordered_map`.
    bool operator==(const ModelVertex &other) const {
        return pos == other.pos
               && texCoord == other.texCoord
               && tangent == other.tangent
               && bitangent == other.bitangent;
    }

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

// as mentioned above, this is implemented to allow using `Vertex` as a key in an `unordered_map`.
template<>
struct std::hash<ModelVertex> {
    size_t operator()(ModelVertex const &vertex) const noexcept {
        return (hash<glm::vec3>()(vertex.pos) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1) ^
               (hash<glm::vec3>()(vertex.normal) << 1) ^
               (hash<glm::vec3>()(vertex.tangent) << 1) ^
               (hash<glm::vec3>()(vertex.bitangent) << 1);
    }
};

struct SkyboxVertex {
    glm::vec3 pos;

    static std::vector<vk::VertexInputBindingDescription> getBindingDescriptions();

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

// vertices of the skybox cube.
// might change this to be generated in a more smart way... but it's good enough for now
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
    {{-1, -1}, {0, 1}},
    {{1, -1}, {1, 1}},
    {{1, 1}, {1, 0}},

    {{-1, -1}, {0, 1}},
    {{1, 1}, {1, 0}},
    {{-1, 1}, {0, 0}},
};
