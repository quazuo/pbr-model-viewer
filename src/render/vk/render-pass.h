#pragma once

#include "src/render/libs.h"
#include "src/render/globals.h"
#include "src/render/renderer.h"

struct RendererContext;

class RenderPass {
    unique_ptr<vk::raii::RenderPass> renderPass;

    friend class RenderPassBuilder;

    RenderPass() = default;

public:
    [[nodiscard]] const vk::raii::RenderPass &operator*() const { return *renderPass; }
};

class RenderPassBuilder {
    std::vector<vk::AttachmentDescription> attachments;

    struct Subpass {
        std::vector<vk::AttachmentReference> colorAttachmentRefs;
        std::vector<vk::AttachmentReference> resolveAttachmentRefs;
        std::optional<vk::AttachmentReference> depthStencilAttachmentRef;
    };

    std::vector<Subpass> subpasses{1};

public:
    RenderPassBuilder &addColorAttachment(const vk::AttachmentDescription &desc);

    RenderPassBuilder &addResolveAttachment(const vk::AttachmentDescription &desc);

    RenderPassBuilder &useDepthStencilAttachment(const vk::AttachmentDescription &desc);

    RenderPassBuilder &beginNewSubpass();

    [[nodiscard]] RenderPass create(const RendererContext &ctx) const;
};
