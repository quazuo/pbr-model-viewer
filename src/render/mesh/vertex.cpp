#include "vertex.h"

std::array<vk::VertexInputBindingDescription, 2> Vertex::getBindingDescription() {
    return {
        {
            {
                .binding = 0u,
                .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
                .inputRate = vk::VertexInputRate::eVertex
            },
            {
                .binding = 1u,
                .stride = static_cast<std::uint32_t>(sizeof(glm::mat4)),
                .inputRate = vk::VertexInputRate::eInstance
            }
        }
    };
}

std::array<vk::VertexInputAttributeDescription, 8> Vertex::getAttributeDescriptions() {
    return {
        {
            {
                .location = 0U,
                .binding = 0U,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = static_cast<std::uint32_t>(offsetof(Vertex, pos)),
            },
            {
                .location = 1U,
                .binding = 0U,
                .format = vk::Format::eR32G32Sfloat,
                .offset = static_cast<std::uint32_t>(offsetof(Vertex, texCoord)),
            },
            {
                .location = 2U,
                .binding = 0U,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = static_cast<std::uint32_t>(offsetof(Vertex, normal)),
            },
            {
                .location = 3U,
                .binding = 0U,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = static_cast<std::uint32_t>(offsetof(Vertex, tangent)),
            },
            {
                .location = 4U,
                .binding = 1U,
                .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = 0U,
            },
            {
                .location = 5U,
                .binding = 1U,
                .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = static_cast<std::uint32_t>(sizeof(glm::vec4)),
            },
            {
                .location = 6U,
                .binding = 1U,
                .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = static_cast<std::uint32_t>(2 * sizeof(glm::vec4)),
            },
            {
                .location = 7U,
                .binding = 1U,
                .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = static_cast<std::uint32_t>(3 * sizeof(glm::vec4)),
            },
        }
    };
}

vk::VertexInputBindingDescription SkyboxVertex::getBindingDescription() {
    return {
        .binding = static_cast<std::uint32_t>(0U),
        .stride = static_cast<std::uint32_t>(sizeof(SkyboxVertex)),
        .inputRate = vk::VertexInputRate::eVertex
    };
}

std::array<vk::VertexInputAttributeDescription, 1> SkyboxVertex::getAttributeDescriptions() {
    return {
        {
            {
                .location = 0U,
                .binding = 0U,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = static_cast<std::uint32_t>(offsetof(SkyboxVertex, pos)),
            },
        }
    };
}
