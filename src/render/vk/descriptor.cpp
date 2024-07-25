#include "descriptor.h"

#include <ranges>

#include "src/render/renderer.h"
#include "buffer.h"
#include "image.h"

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::addBinding(const vk::DescriptorType type, const vk::ShaderStageFlags stages,
                                    const uint32_t descriptorCount) {
    bindings.emplace_back(vk::DescriptorSetLayoutBinding{
        .binding = static_cast<uint32_t>(bindings.size()),
        .descriptorType = type,
        .descriptorCount = descriptorCount,
        .stageFlags = stages,
    });

    return *this;
}

DescriptorLayoutBuilder &
DescriptorLayoutBuilder::addRepeatedBindings(const size_t count, const vk::DescriptorType type,
                                             const vk::ShaderStageFlags stages, const uint32_t descriptorCount) {
    for (size_t i = 0; i < count; i++) {
        addBinding(type, stages, descriptorCount);
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

DescriptorSet &DescriptorSet::queueUpdate(const uint32_t binding, const Buffer &buffer, const vk::DescriptorType type,
                                          const vk::DeviceSize size, const vk::DeviceSize offset,
                                          const uint32_t arrayElement) {
    const vk::DescriptorBufferInfo bufferInfo{
        .buffer = *buffer,
        .offset = offset,
        .range = size,
    };

    queuedUpdates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .arrayElement = arrayElement,
        .type = type,
        .info = bufferInfo,
    });

    return *this;
}

DescriptorSet &DescriptorSet::queueUpdate(const RendererContext &ctx, const uint32_t binding, const Texture &texture,
                                          const uint32_t arrayElement) {
    const vk::DescriptorImageInfo imageInfo{
        .sampler = *texture.getSampler(),
        .imageView = **texture.getImage().getView(ctx),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    queuedUpdates.emplace_back(DescriptorUpdate{
        .binding = binding,
        .arrayElement = arrayElement,
        .type = vk::DescriptorType::eCombinedImageSampler,
        .info = imageInfo,
    });

    return *this;
}

void DescriptorSet::commitUpdates(const RendererContext &ctx) {
    std::vector<vk::WriteDescriptorSet> descriptorWrites;

    for (const auto &update: queuedUpdates) {
        vk::WriteDescriptorSet write{
            .dstSet = **set,
            .dstBinding = update.binding,
            .dstArrayElement = update.arrayElement,
            .descriptorCount = 1,
            .descriptorType = update.type,
        };

        if (std::holds_alternative<vk::DescriptorBufferInfo>(update.info)) {
            write.pBufferInfo = &std::get<vk::DescriptorBufferInfo>(update.info);
        } else if (std::holds_alternative<vk::DescriptorImageInfo>(update.info)) {
            write.pImageInfo = &std::get<vk::DescriptorImageInfo>(update.info);
        } else {
            throw std::runtime_error("unexpected variant in DescriptorSet::commitUpdates");
        }

        descriptorWrites.emplace_back(write);
    }

    ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

    queuedUpdates.clear();
}

void DescriptorSet::updateBinding(const RendererContext &ctx, const uint32_t binding, const Buffer &buffer,
                                  const vk::DescriptorType type, const vk::DeviceSize size,
                                  const vk::DeviceSize offset, const uint32_t arrayElement) const {
    const vk::DescriptorBufferInfo bufferInfo{
        .buffer = *buffer,
        .offset = offset,
        .range = size,
    };

    const vk::WriteDescriptorSet write{
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = arrayElement,
        .descriptorCount = 1,
        .descriptorType = type,
        .pBufferInfo = &bufferInfo,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

void DescriptorSet::updateBinding(const RendererContext &ctx, const uint32_t binding, const Texture &texture,
                                  const uint32_t arrayElement) const {
    const vk::DescriptorImageInfo imageInfo{
        .sampler = *texture.getSampler(),
        .imageView = **texture.getImage().getView(ctx),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet write{
        .dstSet = **set,
        .dstBinding = binding,
        .dstArrayElement = arrayElement,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imageInfo,
    };

    ctx.device->updateDescriptorSets(write, nullptr);
}

std::vector<DescriptorSet>
vkutils::desc::createDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                                    const shared_ptr<vk::raii::DescriptorSetLayout> &layout, const uint32_t count) {
    const std::vector setLayouts(count, **layout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *pool,
        .descriptorSetCount = count,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    std::vector<DescriptorSet> finalSets;

    for (size_t i = 0; i < count; i++) {
        finalSets.emplace_back(layout, std::move(descriptorSets[i]));
    }

    return finalSets;
}
