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
