#include "vertex.h"

vk::VertexInputBindingDescription Vertex::getBindingDescription() {
    return {
        .binding = static_cast<std::uint32_t>(0U),
        .stride = static_cast<std::uint32_t>(sizeof(Vertex)),
        .inputRate = vk::VertexInputRate::eVertex
    };
}

std::array<vk::VertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
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
        }
    };
}
