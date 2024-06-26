#pragma once

#include "src/render/libs.h"
#include "src/render/globals.h"

#include <filesystem>

struct RendererContext;

class Pipeline {
    unique_ptr<vk::raii::Pipeline> pipeline;
    unique_ptr<vk::raii::PipelineLayout> layout;

    friend class PipelineBuilder;

    Pipeline() = default;

public:
    [[nodiscard]]
    const vk::raii::Pipeline& operator*() const { return *pipeline; }

    [[nodiscard]]
    const vk::raii::PipelineLayout& getLayout() const { return *layout; }
};

class PipelineBuilder {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    std::vector<vk::VertexInputBindingDescription> vertexBindings;
    std::vector<vk::VertexInputAttributeDescription> vertexAttributes;

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    uint32_t subpassIndex = 0;

    std::optional<vk::PipelineRasterizationStateCreateInfo> rasterizerOverride;
    std::optional<vk::PipelineMultisampleStateCreateInfo> multisamplingOverride;
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depthStencilOverride;

public:
    PipelineBuilder& withVertexShader(const std::filesystem::path &path);

    PipelineBuilder& withFragmentShader(const std::filesystem::path &path);

    template<typename T>
    PipelineBuilder& withVertices();

    PipelineBuilder &withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts);

    PipelineBuilder &withPushConstants(const std::vector<vk::PushConstantRange> &ranges);

    PipelineBuilder &withRasterizer(const vk::PipelineRasterizationStateCreateInfo& rasterizer);

    PipelineBuilder &withMultisampling(const vk::PipelineMultisampleStateCreateInfo& multisampling);

    PipelineBuilder &withDepthStencil(const vk::PipelineDepthStencilStateCreateInfo& depthStencil);

    PipelineBuilder &forSubpass(uint32_t index);

    [[nodiscard]]
    Pipeline create(const RendererContext &ctx, const vk::raii::RenderPass& renderPass) const;

private:
    void checkParams() const;

    [[nodiscard]]
    static vk::raii::ShaderModule createShaderModule(const RendererContext &ctx, const std::filesystem::path &path);
};
