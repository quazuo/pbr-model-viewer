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

class DescriptorSet {
    unique_ptr<vk::raii::DescriptorSet> set;

    struct DescriptorUpdate {
        uint32_t binding{};
        vk::DescriptorType type{};
        std::variant<vk::DescriptorBufferInfo, vk::DescriptorImageInfo> info;
    };

    std::vector<DescriptorUpdate> queuedUpdates;

public:
    explicit DescriptorSet(vk::raii::DescriptorSet &&s)
        : set(make_unique<vk::raii::DescriptorSet>(std::move(s))) {
    }

    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    DescriptorSet &queueUpdate(uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                               vk::DeviceSize size, vk::DeviceSize offset = 0);

    DescriptorSet &queueUpdate(uint32_t binding, const Texture &texture);

    void commitUpdates(const RendererContext &ctx);

    void updateBinding(const RendererContext &ctx, uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                       vk::DeviceSize size, vk::DeviceSize offset = 0) const;

    void updateBinding(const RendererContext &ctx, uint32_t binding, const Texture &texture) const;
};

namespace utils::desc {
    [[nodiscard]] std::vector<DescriptorSet>
    createDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                         const vk::raii::DescriptorSetLayout &layout, uint32_t count);
}
