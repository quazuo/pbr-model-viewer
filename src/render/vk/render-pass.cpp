#include "render-pass.h"

RenderPassBuilder & RenderPassBuilder::addColorAttachment(const vk::AttachmentDescription &desc) {
    const vk::AttachmentReference ref {
        .attachment = static_cast<uint32_t>(attachments.size()),
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    attachments.emplace_back(desc);
    subpasses.back().colorAttachmentRefs.emplace_back(ref);
    return *this;
}

RenderPassBuilder & RenderPassBuilder::addResolveAttachment(const vk::AttachmentDescription &desc) {
    const vk::AttachmentReference ref {
        .attachment = static_cast<uint32_t>(attachments.size()),
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    attachments.emplace_back(desc);
    subpasses.back().resolveAttachmentRefs.emplace_back(ref);
    return *this;
}

RenderPassBuilder & RenderPassBuilder::useDepthStencilAttachment(const vk::AttachmentDescription &desc) {
    if (subpasses.back().depthStencilAttachmentRef) {
        throw std::invalid_argument("Cannot specify a render pass depth attachment twice!");
    }

    const vk::AttachmentReference ref {
        .attachment = static_cast<uint32_t>(attachments.size()),
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };

    attachments.emplace_back(desc);
    subpasses.back().depthStencilAttachmentRef = ref;
    return *this;
}

RenderPassBuilder & RenderPassBuilder::withSelfDependency(vk::SubpassDependency dependency) {
    dependency.srcSubpass = subpasses.size() - 1;
    dependency.dstSubpass = subpasses.size() - 1;
    subpasses.back().selfDependencies.emplace_back(dependency);
    return *this;
}

RenderPassBuilder & RenderPassBuilder::beginNewSubpass() {
    subpasses.emplace_back();
    return *this;
}

RenderPass RenderPassBuilder::create(const RendererContext &ctx) const {
    RenderPass result;

    std::vector<vk::SubpassDescription> subpassDescriptions;
    std::vector<vk::SubpassDependency> dependencies;

    for (uint32_t i = 0; i < subpasses.size(); i++) {
        const auto& subpass = subpasses[i];

        vk::SubpassDescription desc{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = static_cast<uint32_t>(subpass.colorAttachmentRefs.size()),
            .pColorAttachments = subpass.colorAttachmentRefs.data(),
            .pResolveAttachments = subpass.resolveAttachmentRefs.data(),
        };

        if (subpass.depthStencilAttachmentRef) {
            desc.pDepthStencilAttachment = &subpass.depthStencilAttachmentRef.value();
        }

        const vk::SubpassDependency dependency{
            .srcSubpass = i == 0 ? vk::SubpassExternal : i - 1,
            .dstSubpass = i,
            .srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                             | vk::AccessFlagBits::eDepthStencilAttachmentWrite
        };

        subpassDescriptions.emplace_back(desc);
        dependencies.emplace_back(dependency);
        dependencies.insert(dependencies.end(), subpass.selfDependencies.begin(), subpass.selfDependencies.end());
    }

    const vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = static_cast<uint32_t>(subpassDescriptions.size()),
        .pSubpasses = subpassDescriptions.data(),
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data()
    };

    result.renderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
    return result;
}
