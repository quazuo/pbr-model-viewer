#include "descriptor.h"

#include "src/render/renderer.h"

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
