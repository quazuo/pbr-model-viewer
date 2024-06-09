#include "octree-gen.h"

#include <ctime>

#include "src/render/libs.h"

static std::uint32_t morton3_x(std::uint32_t x) {
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}

static std::uint32_t morton3(const glm::uvec3 coords) {
    const std::uint32_t x = morton3_x(coords.x);
    const std::uint32_t y = morton3_x(coords.y);
    const std::uint32_t z = morton3_x(coords.z);
    return x | (y << 1) | (z << 2);
}

static std::uint32_t unmorton3_x(std::uint32_t x) {
    x &= 0x49249249u;
    x = (x | (x >> 2u))  & 0xc30c30c3u;
    x = (x | (x >> 4u))  & 0x0f00f00fu;
    x = (x | (x >> 8u))  & 0xff0000ffu;
    x = (x | (x >> 16u)) & 0x0000ffffu;
    return x;
}

static glm::uvec3 unmorton3(const std::uint32_t index) {
    const std::uint32_t x = unmorton3_x(index);
    const std::uint32_t y = unmorton3_x(index >> 1u);
    const std::uint32_t z = unmorton3_x(index >> 2u);
    return {x, y, z};
}

constexpr size_t OctreeGen::getOctreeBufferSize(const std::uint32_t depth) {
    std::uint32_t currWidth = 1 << (depth - 1);
    size_t bufferSize = 0;

    while (currWidth > 0) {
        bufferSize += currWidth * currWidth * currWidth;
        currWidth /= 2;
    }

    return depth == 0 ? 1 : bufferSize;
}

constexpr size_t OctreeGen::getOctreeLevelOffset(const std::uint32_t level) {
    if (level == 0) return 0;
    return getOctreeLevelOffset(level - 1) + (1 << (level * 3 - 3));
}

constexpr size_t OctreeGen::getOctreeLevelWidth(const std::uint32_t level) {
    return 1 << level;
}

constexpr size_t OctreeGen::getOctreeLevelSize(const std::uint32_t level) {
    const size_t width = OctreeGen::getOctreeLevelWidth(level);
    return width * width * width;
}

static void makeUpperLevels(const std::uint32_t depth, OctreeGen::OctreeBuf& buf) {
    for (std::uint32_t level = depth - 2; level <= depth - 2; level--) {
        const size_t offset = OctreeGen::getOctreeLevelOffset(level);
        const size_t nextOffset = OctreeGen::getOctreeLevelOffset(level + 1);
        const size_t levelSize = OctreeGen::getOctreeLevelSize(level);

        for (size_t i = 0; i < levelSize; i++) {
            size_t sum = 0;

            for (size_t j = 0; j < 8; j++) {
                sum += buf[nextOffset + 8 * i + j];
            }

            buf[i + offset] = sum == 0 ? 0 : 1;
        }
    }
}

OctreeGen::OctreeBuf OctreeGen::UniformPreset::generate(const std::uint32_t depth) {
    const size_t bufferSize = getOctreeBufferSize(depth);
    OctreeBuf buf(bufferSize, prob >= 1.0f ? 1u : 0u);

    if (prob > 0.0f && prob < 1.0f) {
        srand(time(nullptr));

        const size_t offset = getOctreeLevelOffset(depth - 1);
        const size_t levelSize = getOctreeLevelSize(depth - 1);

        for (int i = 0; i < levelSize; i++) {
            const float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            buf[offset + i] = r < prob ? 1 : 0;
        }
    }

    makeUpperLevels(depth, buf);
    return buf;
}

OctreeGen::OctreeBuf OctreeGen::SpherePreset::generate(const std::uint32_t depth) {
    const size_t bufferSize = getOctreeBufferSize(depth);
    OctreeBuf buf(bufferSize, 0u);

    const size_t offset = getOctreeLevelOffset(depth - 1);
    const size_t levelSize = getOctreeLevelSize(depth - 1);
    const size_t levelWidth = 1 << (depth - 1);
    const auto centerCoords = glm::vec3(static_cast<float>(levelWidth - 1) / 2);

    for (int i = 0; i < levelSize; i++) {
        const glm::uvec3 coords = unmorton3(i);
        buf[offset + i] = glm::length(glm::vec3(coords) - centerCoords) <= radius ? 1 : 0;
    }

    makeUpperLevels(depth, buf);
    return buf;
}

OctreeGen::OctreeBuf OctreeGen::CubePreset::generate(const std::uint32_t depth) {
    const size_t bufferSize = getOctreeBufferSize(depth);
    OctreeBuf buf(bufferSize, 0u);

    const size_t offset = getOctreeLevelOffset(depth - 1);
    const size_t levelSize = getOctreeLevelSize(depth - 1);
    const size_t levelWidth = 1 << (depth - 1);
    const auto centerCoords = glm::vec3(static_cast<float>(levelWidth - 1) / 2);

    for (int i = 0; i < levelSize; i++) {
        const glm::uvec3 coords = unmorton3(i);
        const float parityOffset = sideLength % 2 == 0 ? 0.0f : -0.5f;
        const float maxDist = std::max(
            std::max(
                std::abs(static_cast<float>(coords.x) - centerCoords.x + parityOffset),
                std::abs(static_cast<float>(coords.y) - centerCoords.y + parityOffset)
            ),
            std::abs(static_cast<float>(coords.z) - centerCoords.z + parityOffset)
        );

        buf[offset + i] = maxDist <= static_cast<float>(sideLength) / 2 ? 1 : 0;
    }

    makeUpperLevels(depth, buf);
    return buf;
}
