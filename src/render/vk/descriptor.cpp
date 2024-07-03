#include "descriptor.h"

#include <ranges>

#include "src/render/renderer.h"
#include "buffer.h"
#include "image.h"

DescriptorLayoutBuilder &DescriptorLayoutBuilder::addBinding(vk::DescriptorType type, vk::ShaderStageFlags stages) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = static_cast<uint32_t>(bindings.size()),
        .descriptorType = type,
        .descriptorCount = 1,
        .stageFlags = stages,
    });

    return *this;
}

DescriptorLayoutBuilder &DescriptorLayoutBuilder::addRepeatedBindings(const size_t count, const vk::DescriptorType type,
                                                                      const vk::ShaderStageFlags stages) {
    for (size_t i = 0; i < count; i++) {
        bindings.emplace_back(vk::DescriptorSetLayoutBinding{
            .binding = static_cast<uint32_t>(bindings.size()),
            .descriptorType = type,
            .descriptorCount = 1,
            .stageFlags = stages,
        });
    }

    return *this;
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::create(const RendererContext &ctx) {
    const vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    return {*ctx.device, setLayoutInfo};
}

DescriptorSetBuilder &DescriptorSetBuilder::addBuffer(const Buffer &buffer, const vk::DescriptorType type,
                                                      const vk::DeviceSize size, const vk::DeviceSize offset) {
    const vk::DescriptorBufferInfo bufferInfo{
        .buffer = *buffer,
        .offset = offset,
        .range = size,
    };

    const vk::WriteDescriptorSet descriptorWrite{
        .dstBinding = static_cast<uint32_t>(setWrites.back().size()),
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = type,
    };

    infos.back().emplace_back(bufferInfo);
    setWrites.back().emplace_back(descriptorWrite);
    return *this;
}

DescriptorSetBuilder &DescriptorSetBuilder::addImageSampler(const Texture& texture) {
    const vk::DescriptorImageInfo imageInfo{
        .sampler = texture.getSampler(),
        .imageView = texture.getView(),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet descriptorWrite{
        .dstBinding = static_cast<uint32_t>(setWrites.back().size()),
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
    };

    infos.back().emplace_back(imageInfo);
    setWrites.back().emplace_back(descriptorWrite);
    return *this;
}

DescriptorSetBuilder &DescriptorSetBuilder::beginNewSet() {
    infos.emplace_back();
    setWrites.emplace_back();
    return *this;
}

std::vector<vk::raii::DescriptorSet>
DescriptorSetBuilder::create(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                             const vk::raii::DescriptorSetLayout &layout) {
    const uint32_t setsCount = setWrites.size();
    const std::vector setLayouts(setsCount, *layout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *pool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        for (size_t j = 0; j < setWrites[i].size(); j++) {
            setWrites[i][j].dstSet = *descriptorSets[i];

            if (std::holds_alternative<vk::DescriptorBufferInfo>(infos[i][j])) {
                setWrites[i][j].pBufferInfo = &std::get<vk::DescriptorBufferInfo>(infos[i][j]);
            } else if (std::holds_alternative<vk::DescriptorImageInfo>(infos[i][j])) {
                setWrites[i][j].pImageInfo = &std::get<vk::DescriptorImageInfo>(infos[i][j]);
            }
        }

        ctx.device->updateDescriptorSets(setWrites[i], nullptr);
    }

    return descriptorSets;
}
