#pragma once

#include <filesystem>
#include <vector>

struct Vertex;

class Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

public:
    explicit Mesh(const std::filesystem::path& path);

    [[nodiscard]]
    const std::vector<Vertex>& getVertices() const { return vertices; }

    [[nodiscard]]
    const std::vector<std::uint32_t>& getIndices() const { return indices; }
};
