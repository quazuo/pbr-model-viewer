#pragma once

#include <vector>

namespace OctreeGen {
    using OctreeBuf = std::vector<std::uint8_t>;

    constexpr size_t getOctreeBufferSize(std::uint32_t depth);

    constexpr size_t getOctreeLevelOffset(std::uint32_t level);

    constexpr size_t getOctreeLevelWidth(std::uint32_t level);

    constexpr size_t getOctreeLevelSize(std::uint32_t level);

    class Preset {
    public:
        virtual ~Preset() = default;

        [[nodiscard]]
        virtual OctreeBuf generate(std::uint32_t depth) = 0;
    };

    class UniformPreset final : public Preset {
        float prob;

    public:
        explicit UniformPreset(const float p) : prob(p) {}

        OctreeBuf generate(std::uint32_t depth) override;
    };

    class SpherePreset final : public Preset {
        float radius;

    public:
        explicit SpherePreset(const float r) : radius(r) {}

        OctreeBuf generate(std::uint32_t depth) override;
    };

    class CubePreset final : public Preset {
        std::uint32_t sideLength;

    public:
        explicit CubePreset(const std::uint32_t sl) : sideLength(sl) {}

        OctreeBuf generate(std::uint32_t depth) override;
    };
}
