#pragma once

#include "src/render/libs.h"
#include "src/render/globals.h"

struct RendererContext;

class DescriptorLayoutBuilder {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

public:
    DescriptorLayoutBuilder& addBinding(vk::DescriptorType type, vk::ShaderStageFlags stages);

    DescriptorLayoutBuilder& addRepeatedBindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext& ctx);
};
