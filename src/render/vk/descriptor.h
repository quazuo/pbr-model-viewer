#pragma once

#include <utility>
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

    DescriptorLayoutBuilder &addArrayBinding(vk::DescriptorType type, vk::ShaderStageFlags stages, uint32_t size);

    DescriptorLayoutBuilder &addRepeatedBindings(size_t count, vk::DescriptorType type, vk::ShaderStageFlags stages);

    DescriptorLayoutBuilder &addRepeatedArrayBindings(size_t count, vk::DescriptorType type,
                                                      vk::ShaderStageFlags stages, uint32_t size);

    [[nodiscard]] vk::raii::DescriptorSetLayout create(const RendererContext &ctx);
};

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

    DescriptorSet &queueUpdate(uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                               vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t arrayElement = 0);

    DescriptorSet &queueUpdate(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                               uint32_t arrayElement = 0);

    void commitUpdates(const RendererContext &ctx);

    void updateBinding(const RendererContext &ctx, uint32_t binding, const Buffer &buffer, vk::DescriptorType type,
                       vk::DeviceSize size, vk::DeviceSize offset = 0, uint32_t arrayElement = 0) const;

    void updateBinding(const RendererContext &ctx, uint32_t binding, const Texture &texture,
                       uint32_t arrayElement = 0) const;
};

namespace vkutils::desc {
    [[nodiscard]] std::vector<DescriptorSet>
    createDescriptorSets(const RendererContext &ctx, const vk::raii::DescriptorPool &pool,
                         const shared_ptr<vk::raii::DescriptorSetLayout> &layout, uint32_t count);
}
