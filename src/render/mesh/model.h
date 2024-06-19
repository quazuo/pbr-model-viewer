#pragma once

#include <filesystem>
#include <vector>

struct Vertex;
struct aiScene;
struct aiMesh;
struct aiNode;

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    explicit Mesh(const aiMesh* assimpMesh);
};

class Model {
    std::vector<Mesh> meshes;

public:
    explicit Model(const std::filesystem::path& path);

    [[nodiscard]]
    const std::vector<Mesh>& getMeshes() const { return meshes; }

    [[nodiscard]]
    std::vector<Vertex> getVertices() const;

    [[nodiscard]]
    std::vector<std::uint32_t> getIndices() const;
};
