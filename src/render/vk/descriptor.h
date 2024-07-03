#pragma once

#include <variant>

#include "src/render/libs.h"
#include "src/render/globals.h"

struct RendererContext;
class Buffer;
class Texture;

class DescriptorLayoutBuilder {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

public:
    DescriptorLayoutBuilder &addBinding(vk::DescriptorType type, vk::ShaderStageFlags stages);

    DescriptorLayoutBuilder &addRepeatedBindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext &ctx);
};

class DescriptorSetBuilder {
    using DescriptorInfo = std::variant<vk::DescriptorBufferInfo, vk::DescriptorImageInfo>;

    std::vector<std::vector<DescriptorInfo>> infos{1};
    std::vector<std::vector<vk::WriteDescriptorSet> > setWrites{1};

public:
    DescriptorSetBuilder &addBuffer(const Buffer &buffer, vk::DescriptorType type, vk::DeviceSize size,
                                    vk::DeviceSize offset = 0);

    DescriptorSetBuilder &addImageSampler(const Texture& texture);

    DescriptorSetBuilder &beginNewSet();

    [[nodiscard]] std::vector<vk::raii::DescriptorSet>
    create(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
           const vk::raii::DescriptorSetLayout &layout);
};
