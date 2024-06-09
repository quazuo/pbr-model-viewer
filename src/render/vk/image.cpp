#include "image.h"

#include <filesystem>

#include "deps/stb/stb_image.h"

#include "buffer.h"
#include "cmd.h"
#include "src/render/renderer.h"

Image::Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
             const vk::MemoryPropertyFlags properties)
    : allocator(ctx.allocator->get()), extent(imageInfo.extent) {
    const VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    VkImage newImage;
    VmaAllocation newAllocation;

    const auto result = vmaCreateImage(
        allocator,
        reinterpret_cast<const VkImageCreateInfo *>(&imageInfo),
        &allocInfo,
        &newImage,
        &newAllocation,
        nullptr
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer!");
    }

    image = std::make_unique<vk::raii::Image>(*ctx.device, newImage);
    allocation = std::make_unique<VmaAllocation>(newAllocation);
}

Image::~Image() {
    vmaFreeMemory(allocator, *allocation);
}

const vk::raii::ImageView &Image::getView() const {
    if (!view) {
        throw std::runtime_error("tried to acquire image view without creating it!");
    }

    return *view;
}

void Image::createView(const RendererContext &ctx, const vk::Format format, const vk::ImageAspectFlags aspectFlags,
                       const std::uint32_t mipLevels) {
    view = utils::img::createImageView(ctx, **image, format, aspectFlags, mipLevels);
}

void Image::copyFromBuffer(const RendererContext &ctx, const vk::Buffer buffer, const vk::raii::CommandPool &cmdPool,
                           const vk::raii::Queue &queue) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0U,
        .bufferRowLength = 0U,
        .bufferImageHeight = 0U,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = extent,
    };

    utils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        cmdBuffer.copyBufferToImage(
            buffer,
            **image,
            vk::ImageLayout::eTransferDstOptimal,
            region
        );
    });
}

std::unique_ptr<vk::raii::ImageView>
utils::img::createImageView(const RendererContext &ctx, const vk::Image image, const vk::Format format,
                            const vk::ImageAspectFlags aspectFlags, const std::uint32_t mipLevels) {
    const vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    return std::make_unique<vk::raii::ImageView>(*ctx.device, createInfo);
}

void utils::img::transitionImageLayout(const RendererContext &ctx, const vk::Image image,
                                       const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                                       const std::uint32_t mipLevels, const vk::raii::CommandPool &cmdPool,
                                       const vk::raii::Queue &queue) {
    vk::AccessFlags srcAccessMask, dstAccessMask;
    vk::PipelineStageFlags sourceStage, destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        srcAccessMask = {};
        dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal
               && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eGeneral) {
        srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage = vk::PipelineStageFlagBits::eComputeShader,
                destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        cmdBuffer.pipelineBarrier(
            sourceStage, destinationStage,
            {},
            nullptr,
            nullptr,
            barrier
        );
    });
}
