#pragma once

#include <filesystem>
#include <vector>

#include "vertex.h"
#include "src/render/libs.h"
#include "src/render/globals.h"

struct RendererContext;
struct aiMaterial;
struct aiScene;
struct aiMesh;
struct aiNode;
class DescriptorSet;
class Texture;

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<glm::mat4> instances;
    uint32_t materialID;

    explicit Mesh(const aiMesh *assimpMesh);
};

struct Material {
    unique_ptr<Texture> baseColor;
    unique_ptr<Texture> normal;
    unique_ptr<Texture> orm;

    Material() = default;

    explicit Material(const RendererContext &ctx, const aiMaterial *assimpMaterial,
                      const std::filesystem::path &basePath);
};

class Model {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;

public:
    explicit Model(const RendererContext &ctx, const std::filesystem::path &path, bool loadMaterials);

    void addInstances(const aiNode *node, const glm::mat4 &baseTransform);

    [[nodiscard]] const std::vector<Mesh> &getMeshes() const { return meshes; }

    [[nodiscard]] const std::vector<Material> &getMaterials() const { return materials; }

    [[nodiscard]] std::vector<Vertex> getVertices() const;

    [[nodiscard]] std::vector<uint32_t> getIndices() const;

    [[nodiscard]] std::vector<glm::mat4> getInstanceTransforms() const;

private:
    void normalizeScale();

    [[nodiscard]] float getMaxVertexDistance() const;
};
