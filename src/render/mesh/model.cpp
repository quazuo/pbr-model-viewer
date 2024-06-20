#include "model.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "vertex.h"

Mesh::Mesh(const aiMesh *assimpMesh) {
    std::unordered_map<Vertex, std::uint32_t> uniqueVertices;

    for (size_t faceIdx = 0; faceIdx < assimpMesh->mNumFaces; faceIdx++) {
        const auto& face = assimpMesh->mFaces[faceIdx];

        for (size_t i = 0; i < face.mNumIndices; i++) {
            const Vertex vertex{
                .pos = {
                    assimpMesh->mVertices[face.mIndices[i]].x,
                    assimpMesh->mVertices[face.mIndices[i]].y,
                    assimpMesh->mVertices[face.mIndices[i]].z
                },
                .texCoord = {
                    assimpMesh->mTextureCoords[0][face.mIndices[i]].x,
                    1.0f - assimpMesh->mTextureCoords[0][face.mIndices[i]].y
                },
                .normal = {
                    assimpMesh->mNormals[face.mIndices[i]].x,
                    assimpMesh->mNormals[face.mIndices[i]].y,
                    assimpMesh->mNormals[face.mIndices[i]].z
                }
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

    addInstances(scene->mRootNode);
}

static inline glm::mat4 assimpMatrixToGlm(const aiMatrix4x4& m) {
    glm::mat4 res;

    //the a,b,c,d in assimp is the row ; the 1,2,3,4 is the column
    res[0][0] = m.a1; res[1][0] = m.a2; res[2][0] = m.a3; res[3][0] = m.a4;
    res[0][1] = m.b1; res[1][1] = m.b2; res[2][1] = m.b3; res[3][1] = m.b4;
    res[0][2] = m.c1; res[1][2] = m.c2; res[2][2] = m.c3; res[3][2] = m.c4;
    res[0][3] = m.d1; res[1][3] = m.d2; res[2][3] = m.d3; res[3][3] = m.d4;

    return res;
}

void Model::addInstances(const aiNode *node) {
    for (size_t i = 0; i < node->mNumMeshes; i++) {
        meshes[node->mMeshes[i]].instances.push_back(assimpMatrixToGlm(node->mTransformation));
    }

    for (size_t i = 0; i < node->mNumChildren; i++) {
        addInstances(node->mChildren[i]);
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

std::vector<std::uint32_t> Model::getIndices() const {
    std::vector<std::uint32_t> indices;

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
