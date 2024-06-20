#pragma once

#include <filesystem>
#include <vector>

#include "vertex.h"
#include "src/render/libs.h"

struct aiScene;
struct aiMesh;
struct aiNode;

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<glm::mat4> instances;

    explicit Mesh(const aiMesh* assimpMesh);
};

class Model {
    std::vector<Mesh> meshes;

public:
    explicit Model(const std::filesystem::path& path);

    void addInstances(const aiNode* node);

    [[nodiscard]]
    const std::vector<Mesh>& getMeshes() const { return meshes; }

    [[nodiscard]]
    std::vector<Vertex> getVertices() const;

    [[nodiscard]]
    std::vector<std::uint32_t> getIndices() const;

    [[nodiscard]]
    std::vector<glm::mat4> getInstanceTransforms() const;
};
