#include "model.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "vertex.h"

static glm::vec3 assimpVecToGlm(const aiVector3D& v) {
    return { v.x, v.y, v.z };
}

static glm::mat4 assimpMatrixToGlm(const aiMatrix4x4& m) {
    glm::mat4 res;

    res[0][0] = m.a1; res[1][0] = m.a2; res[2][0] = m.a3; res[3][0] = m.a4;
    res[0][1] = m.b1; res[1][1] = m.b2; res[2][1] = m.b3; res[3][1] = m.b4;
    res[0][2] = m.c1; res[1][2] = m.c2; res[2][2] = m.c3; res[3][2] = m.c4;
    res[0][3] = m.d1; res[1][3] = m.d2; res[2][3] = m.d3; res[3][3] = m.d4;

    return res;
}

Mesh::Mesh(const aiMesh *assimpMesh) {
    std::unordered_map<Vertex, uint32_t> uniqueVertices;

    for (size_t faceIdx = 0; faceIdx < assimpMesh->mNumFaces; faceIdx++) {
        const auto& face = assimpMesh->mFaces[faceIdx];

        for (size_t i = 0; i < face.mNumIndices; i++) {
            const Vertex vertex{
                .pos = assimpVecToGlm(assimpMesh->mVertices[face.mIndices[i]]),
                .texCoord = {
                    assimpMesh->mTextureCoords[0][face.mIndices[i]].x,
                    1.0f - assimpMesh->mTextureCoords[0][face.mIndices[i]].y
                },
                .normal = assimpVecToGlm(assimpMesh->mNormals[face.mIndices[i]]),
                .tangent = assimpVecToGlm(assimpMesh->mTangents[face.mIndices[i]]),
            };

            if (!uniqueVertices.contains(vertex)) {
                uniqueVertices[vertex] = vertices.size();
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices.at(vertex));
        }
    }
}

Model::Model(const std::filesystem::path &path) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(path.string(),
        aiProcess_CalcTangentSpace       |
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

    if (!scene) {
        throw std::runtime_error(importer.GetErrorString());
    }

    for (size_t i = 0; i < scene->mNumMeshes; i++) {
        meshes.emplace_back(scene->mMeshes[i]);
    }

    addInstances(scene->mRootNode, glm::identity<glm::mat4>());

    normalizeScale();
}

void Model::addInstances(const aiNode *node, const glm::mat4& baseTransform) {
    const glm::mat4 transform = baseTransform * assimpMatrixToGlm(node->mTransformation);

    for (size_t i = 0; i < node->mNumMeshes; i++) {
        meshes[node->mMeshes[i]].instances.push_back(transform);
    }

    for (size_t i = 0; i < node->mNumChildren; i++) {
        addInstances(node->mChildren[i], transform);
    }
}

std::vector<Vertex> Model::getVertices() const {
    std::vector<Vertex> vertices;

    size_t totalSize = 0;
    for (const auto& mesh : meshes) {
        totalSize += mesh.vertices.size();
    }

    vertices.reserve(totalSize);

    for (const auto& mesh : meshes) {
        vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    }

    return vertices;
}

std::vector<uint32_t> Model::getIndices() const {
    std::vector<uint32_t> indices;

    size_t totalSize = 0;
    for (const auto& mesh : meshes) {
        totalSize += mesh.indices.size();
    }

    indices.reserve(totalSize);

    for (const auto& mesh : meshes) {
        indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
    }

    return indices;
}

std::vector<glm::mat4> Model::getInstanceTransforms() const {
    std::vector<glm::mat4> result;

    size_t totalSize = 0;
    for (const auto& mesh : meshes) {
        totalSize += mesh.instances.size();
    }

    result.reserve(totalSize);

    for (const auto& mesh : meshes) {
        result.insert(result.end(), mesh.instances.begin(), mesh.instances.end());
    }

    return result;
}

void Model::normalizeScale() {
    const float largestDistance = getMaxVertexDistance();
    const glm::mat4 scaleMatrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3(1.0f / largestDistance));

    for (auto& mesh: meshes) {
        for (auto& transform: mesh.instances) {
            transform = scaleMatrix * transform;
        }
    }
}

float Model::getMaxVertexDistance() const {
    float largestDistance = 0.0;

    for (const auto& mesh: meshes) {
        for (const auto& vertex: mesh.vertices) {
            for (const auto& transform: mesh.instances) {
                largestDistance = std::max(
                    largestDistance,
                    glm::length(glm::vec3(transform * glm::vec4(vertex.pos, 1.0)))
                );
            }
        }
    }

    return largestDistance;
}
