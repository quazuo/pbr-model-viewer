#pragma once

enum class FileType {
    MODEL,
    ALBEDO_PNG,
    NORMAL_PNG,
    ORM_PNG,
    RMA_PNG,
    AO_PNG,
    ROUGHNESS_PNG,
    METALLIC_PNG,
    ENVMAP_HDR,
};

[[nodiscard]] static std::vector<std::string> getFileTypeExtensions(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return {".obj", ".fbx"};
        case FileType::ALBEDO_PNG:
        case FileType::NORMAL_PNG:
        case FileType::ORM_PNG:
        case FileType::RMA_PNG:
        case FileType::AO_PNG:
        case FileType::ROUGHNESS_PNG:
        case FileType::METALLIC_PNG:
            return {".png"};
        case FileType::ENVMAP_HDR:
            return {".hdr"};
        default:
            throw std::runtime_error("unexpected filetype in getFileTypeExtensions");
    }
}

[[nodiscard]] static bool isFileTypeOptional(const FileType type) {
    switch (type) {
        case FileType::AO_PNG:
        case FileType::METALLIC_PNG:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] static std::string getFileTypeLoadLabel(const FileType type) {
    switch (type) {
        case FileType::MODEL:
            return "Load model...";
        case FileType::ALBEDO_PNG:
            return "Load color texture...";
        case FileType::NORMAL_PNG:
            return "Load normal map...";
        case FileType::ORM_PNG:
            return "Load ORM map...";
        case FileType::RMA_PNG:
            return "Load RMA map...";
        case FileType::AO_PNG:
            return "Load AO map...";
        case FileType::ROUGHNESS_PNG:
            return "Load roughness map...";
        case FileType::METALLIC_PNG:
            return "Load metallic map...";
        case FileType::ENVMAP_HDR:
            return "Load environment map...";
        default:
            throw std::runtime_error("unexpected filetype in getFileTypeLoadLabel");
    }
}

struct FileLoadScheme {
    std::string name;
    std::set<FileType> requirements;
};

static const std::vector<FileLoadScheme> fileLoadSchemes{
    {
        "Albedo + Normal + ORM",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::ORM_PNG,
        }
    },
    {
        "Albedo + Normal + RMA",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::RMA_PNG,
        }
    },
    {
        "Albedo + Normal + AO + Roughness + Metallic",
        {
            FileType::MODEL,
            FileType::ALBEDO_PNG,
            FileType::NORMAL_PNG,
            FileType::AO_PNG,
            FileType::ROUGHNESS_PNG,
            FileType::METALLIC_PNG
        }
    },
};
