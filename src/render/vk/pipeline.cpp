#include "pipeline.h"

#include <fstream>

#include "src/render/renderer.h"
#include "src/render/mesh/vertex.h"

PipelineBuilder &PipelineBuilder::withVertexShader(const std::filesystem::path &path) {
    vertexShaderPath = path;
    return *this;
}

PipelineBuilder &PipelineBuilder::withFragmentShader(const std::filesystem::path &path) {
    fragmentShaderPath = path;
    return *this;
}

template<typename T>
PipelineBuilder &PipelineBuilder::withVertices() {
    vertexBindings = T::getBindingDescriptions();
    vertexAttributes = T::getAttributeDescriptions();
    return *this;
}

PipelineBuilder &PipelineBuilder::withDescriptorLayouts(const std::vector<vk::DescriptorSetLayout> &layouts) {
    descriptorSetLayouts = layouts;
    return *this;
}

PipelineBuilder & PipelineBuilder::withPushConstants(const std::vector<vk::PushConstantRange> &ranges) {
    pushConstantRanges = ranges;
    return *this;
}

PipelineBuilder &PipelineBuilder::withRasterizer(const vk::PipelineRasterizationStateCreateInfo &rasterizer) {
    rasterizerOverride = rasterizer;
    return *this;
}

PipelineBuilder &PipelineBuilder::withMultisampling(const vk::PipelineMultisampleStateCreateInfo &multisampling) {
    multisamplingOverride = multisampling;
    return *this;
}

PipelineBuilder &PipelineBuilder::withDepthStencil(const vk::PipelineDepthStencilStateCreateInfo &depthStencil) {
    depthStencilOverride = depthStencil;
    return *this;
}

PipelineBuilder &PipelineBuilder::forSubpass(const uint32_t index) {
    subpassIndex = index;
    return *this;
}

Pipeline PipelineBuilder::create(const RendererContext &ctx, const vk::raii::RenderPass &renderPass) const {
    Pipeline result;

    vk::raii::ShaderModule vertShaderModule = createShaderModule(ctx, vertexShaderPath);
    vk::raii::ShaderModule fragShaderModule = createShaderModule(ctx, fragmentShaderPath);

    const vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = *vertShaderModule,
        .pName = "main",
    };

    const vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = *fragShaderModule,
        .pName = "main",
    };

    const std::vector shaderStages{
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindings.size()),
        .pVertexBindingDescriptions = vertexBindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size()),
        .pVertexAttributeDescriptions = vertexAttributes.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    static constexpr std::array dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    static constexpr vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    constexpr vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1U,
        .scissorCount = 1U,
    };

    const auto rasterizer = rasterizerOverride
                                ? *rasterizerOverride
                                : vk::PipelineRasterizationStateCreateInfo{
                                    .polygonMode = vk::PolygonMode::eFill,
                                    .cullMode = vk::CullModeFlagBits::eBack,
                                    .frontFace = vk::FrontFace::eCounterClockwise,
                                    .lineWidth = 1.0f,
                                };

    const auto multisampling = multisamplingOverride
                                   ? *multisamplingOverride
                                   : vk::PipelineMultisampleStateCreateInfo{
                                       .rasterizationSamples = vk::SampleCountFlagBits::e1,
                                       .minSampleShading = 1.0f,
                                   };

    static constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA,
    };

    static constexpr vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .attachmentCount = 1u,
        .pAttachments = &colorBlendAttachment,
    };

    const auto depthStencil = depthStencilOverride
                                  ? *depthStencilOverride
                                  : vk::PipelineDepthStencilStateCreateInfo{
                                      .depthTestEnable = vk::True,
                                      .depthWriteEnable = vk::True,
                                      .depthCompareOp = vk::CompareOp::eLess,
                                  };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
        .pPushConstantRanges = pushConstantRanges.empty() ? nullptr : pushConstantRanges.data()
    };

    result.layout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = **result.layout,
        .renderPass = *renderPass,
        .subpass = subpassIndex,
    };

    result.pipeline = make_unique<vk::raii::Pipeline>(*ctx.device, nullptr, pipelineInfo);

    return result;
}

void PipelineBuilder::checkParams() const {
    if (vertexShaderPath.empty()) {
        throw std::invalid_argument("vertex shader must be specified during pipeline creation!");
    }

    if (fragmentShaderPath.empty()) {
        throw std::invalid_argument("fragment shader must be specified during pipeline creation!");
    }

    if (vertexBindings.empty() && vertexAttributes.empty()) {
        throw std::invalid_argument("vertex descriptions must be specified during pipeline creation!");
    }
}

vk::raii::ShaderModule
PipelineBuilder::createShaderModule(const RendererContext &ctx, const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    const size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    const vk::ShaderModuleCreateInfo createInfo{
        .codeSize = buffer.size(),
        .pCode = reinterpret_cast<const uint32_t *>(buffer.data()),
    };

    return vk::raii::ShaderModule{*ctx.device, createInfo};
}

template PipelineBuilder &PipelineBuilder::withVertices<Vertex>();
template PipelineBuilder &PipelineBuilder::withVertices<SkyboxVertex>();
