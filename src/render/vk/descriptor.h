#pragma once

#include <utility>
#include <variant>

#include "src/render/libs.h"
#include "src/render/globals.h"

struct RendererContext;
class Buffer;
class Texture;

/**
 * Builder class streamlining descriptor set layout creation.
 *
 * Methods which add bindings are order-dependent and the order in which they are called
 * defines which binding is used for a given resource, i.e. first call to `addBinding` will
 * use binding 0, second will use binding 1, and so on.
 */
class DescriptorLayoutBuilder {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

public:
    DescriptorLayoutBuilder &addBinding(vk::DescriptorType type, vk::ShaderStageFlags stages,
                                        uint32_t descriptorCount = 1);

    DescriptorLayoutBuilder &addRepeatedBindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages,
                                                 uint32_t descriptorCount = 1);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext &ctx);
};

/**
 * Convenience wrapper around Vulkan descriptor sets, mainly to pair them together with related layouts,
 * as well as provide an easy way to update them in a performant way.
 */
class DescriptorSet {
    shared_ptr<vk::raii::DescriptorSetLayout> layout;
    unique_ptr<vk::raii::DescriptorSet> set;

    struct DescriptorUpdate {
        uint32_t binding{};
        uint32_t arrayElement{};
        vk::DescriptorType type{};
        std::variant<vk::DescriptorBufferInfo, vk::DescriptorImageInfo> info;
    };

    std::vector<DescriptorUpdate> queuedUpdates;

public:
    explicit DescriptorSet(decltype(layout) l, vk::raii::DescriptorSet &&s)
        : layout(std::move(l)), set(make_unique<vk::raii::DescriptorSet>(std::move(s))) {
    }

    [[nodiscard]] const vk::raii::DescriptorSet &operator*() const { return *set; }

    [[nodiscard]] const vk::raii::DescriptorSetLayout &getLayout() const { return *layout; }

    /**
     * Queues an update to a given binding in this descriptor set.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queueUpdate(uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                               vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t arrayElement = 0);

    /**
     * Queues an update to a given binding in this descriptor set.
     * To actually push the update, `commitUpdates` must be called after all desired updates are queued.
     */
    DescriptorSet &queueUpdate(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                               uint32_t arrayElement = 0);

    void commitUpdates(const RendererContext &ctx);

    /**
     * Immediately updates a single binding in this descriptor set.
     */
    void updateBinding(const RendererContext &ctx, uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                       vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t arrayElement = 0) const;

    /**
     * Immediately updates a single binding in this descriptor set.
     */
    void updateBinding(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                       uint32_t arrayElement = 0) const;
};

namespace vkutils::desc {
    [[nodiscard]] std::vector<DescriptorSet>
    createDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                         const shared_ptr<vk::raii::DescriptorSetLayout> &layout, uint32_t count);
}
