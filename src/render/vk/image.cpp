#include "image.h"

#include <filesystem>
#include <map>

#include "deps/stb/stb_image.h"
#include "deps/stb/stb_image_write.h"

#include "buffer.h"
#include "cmd.h"
#include "src/render/renderer.h"

Image::Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
             const vk::MemoryPropertyFlags properties, const vk::ImageAspectFlags aspect)
    : allocator(**ctx.allocator),
      extent(imageInfo.extent),
      format(imageInfo.format),
      mipLevels(imageInfo.mipLevels),
      aspectMask(aspect) {
    VmaAllocationCreateFlags flags;
    if (properties & vk::MemoryPropertyFlagBits::eDeviceLocal) {
        flags = 0;
    } else {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    const VmaAllocationCreateInfo allocInfo{
        .flags = flags,
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

    image = make_unique<vk::raii::Image>(*ctx.device, newImage);
    allocation = make_unique<VmaAllocation>(newAllocation);
}

Image::~Image() {
    vmaFreeMemory(allocator, *allocation);
}

void Image::createViews(const RendererContext &ctx) {
    view = vkutils::img::createImageView(ctx, **image, format, aspectMask, 0, mipLevels, 0);
    attachmentView = vkutils::img::createImageView(ctx, **image, format, aspectMask, 0, 1, 0);

    for (uint32_t mip = 0; mip < mipLevels; mip++) {
        mipViews.emplace_back(
            vkutils::img::createImageView(ctx, **image, format, aspectMask, mip, 1, 0)
        );
    }
}

void Image::copyFromBuffer(const vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer) {
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

    commandBuffer.copyBufferToImage(
        buffer,
        **image,
        vk::ImageLayout::eTransferDstOptimal,
        region
    );
}

void Image::transitionLayout(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                             const vk::raii::CommandBuffer &commandBuffer) const {
    const vk::ImageSubresourceRange range{
        .aspectMask = aspectMask,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    transitionLayout(oldLayout, newLayout, range, commandBuffer);
}

void Image::transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                             vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &commandBuffer) const {
    if (!transitionBarrierSchemes.contains({oldLayout, newLayout})) {
        throw std::invalid_argument("unsupported layout transition!");
    }

    const auto &[srcAccessMask, dstAccessMask, srcStage, dstStage] = transitionBarrierSchemes[{oldLayout, newLayout}];

    range.aspectMask = aspectMask;

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = srcAccessMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = **image,
        .subresourceRange = range,
    };

    commandBuffer.pipelineBarrier(
        srcStage,
        dstStage,
        {},
        nullptr,
        nullptr,
        barrier
    );
}

void Image::saveToFile(const RendererContext &ctx, const std::filesystem::path &path,
                       const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue) const {
    const vk::ImageCreateInfo tempImageInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eLinear,
        .usage = vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const Image tempImage{
        ctx,
        tempImageInfo,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::ImageAspectFlagBits::eColor
    };

    vkutils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        transitionLayout(
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            cmdBuffer
        );

        transitionLayout(
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            cmdBuffer
        );
    });

    const vk::ImageCopy imageCopyRegion{
        .srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .dstSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .extent = extent
    };

    const vk::Offset3D blitOffset{
        .x = static_cast<int32_t>(extent.width),
        .y = static_cast<int32_t>(extent.height),
        .z = static_cast<int32_t>(extent.depth)
    };

    const vk::ImageMemoryBarrier2 imageMemoryBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = **tempImage,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        }
    };

    const vk::DependencyInfo dependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &imageMemoryBarrier
    };

    const vk::ImageBlit blitInfo{
        .srcSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .srcOffsets = {{vk::Offset3D(), blitOffset}},
        .dstSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .layerCount = 1
        },
        .dstOffsets = {{vk::Offset3D(), blitOffset}}
    };

    bool supportsBlit = true;

    // check if the device supports blitting from this image's format
    const vk::FormatProperties srcFormatProperties = ctx.physicalDevice->getFormatProperties(format);
    if (!(srcFormatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc)) {
        supportsBlit = false;
    }

    // check if the device supports blitting to linear images
    const vk::FormatProperties dstFormatProperties = ctx.physicalDevice->getFormatProperties(tempImage.format);
    if (!(dstFormatProperties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
        supportsBlit = false;
    }

    vkutils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &commandBuffer) {
        if (supportsBlit) {
            commandBuffer.blitImage(
                **image,
                vk::ImageLayout::eTransferSrcOptimal,
                **tempImage,
                vk::ImageLayout::eTransferDstOptimal,
                blitInfo,
                vk::Filter::eLinear
            );
        } else {
            commandBuffer.copyImage(
                **image,
                vk::ImageLayout::eTransferSrcOptimal,
                **tempImage,
                vk::ImageLayout::eTransferDstOptimal,
                imageCopyRegion
            );
        }

        commandBuffer.pipelineBarrier2(dependencyInfo);
    });

    void *data;
    vmaMapMemory(tempImage.allocator, *tempImage.allocation, &data);

    stbi_write_png(
        path.string().c_str(),
        static_cast<int>(tempImage.extent.width),
        static_cast<int>(tempImage.extent.height),
        STBI_rgb_alpha,
        data,
        vkutils::img::getFormatSizeInBytes(tempImage.format) * tempImage.extent.width
    );

    vmaUnmapMemory(tempImage.allocator, *tempImage.allocation);

    vkutils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        transitionLayout(
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            cmdBuffer
        );
    });
}

// ==================== CubeImage ====================

CubeImage::CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                     const vk::MemoryPropertyFlags properties)
    : Image(ctx, imageInfo, properties, vk::ImageAspectFlagBits::eColor) {
}

void CubeImage::createViews(const RendererContext &ctx) {
    view = vkutils::img::createCubeImageView(ctx, **image, format, aspectMask, 0, mipLevels);
    attachmentView = vkutils::img::createCubeImageView(ctx, **image, format, aspectMask, 0, 1);

    for (uint32_t mip = 0; mip < mipLevels; mip++) {
        mipViews.emplace_back(
            vkutils::img::createCubeImageView(ctx, **image, format, aspectMask, mip, 1)
        );
    }

    for (uint32_t layer = 0; layer < 6; layer++) {
        layerViews.emplace_back(
            vkutils::img::createImageView(ctx, **image, format, aspectMask, 0, mipLevels, layer)
        );
        attachmentLayerViews.emplace_back(
            vkutils::img::createImageView(ctx, **image, format, aspectMask, 0, 1, layer)
        );

        std::vector<unique_ptr<vk::raii::ImageView> > currLayerMipViews;

        for (uint32_t mip = 0; mip < mipLevels; mip++) {
            currLayerMipViews.emplace_back(
                vkutils::img::createImageView(ctx, **image, format, aspectMask, mip, 1, layer)
            );
        }

        layerMipViews.emplace_back(std::move(currLayerMipViews));
    }
}

void CubeImage::copyFromBuffer(const vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0U,
        .bufferRowLength = 0U,
        .bufferImageHeight = 0U,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 6,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = extent,
    };

    commandBuffer.copyBufferToImage(
        buffer,
        **image,
        vk::ImageLayout::eTransferDstOptimal,
        region
    );
}

void CubeImage::transitionLayout(const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                                 const vk::raii::CommandBuffer &commandBuffer) const {
    const vk::ImageSubresourceRange range{
        .aspectMask = aspectMask,
        .baseMipLevel = 0,
        .levelCount = mipLevels,
        .baseArrayLayer = 0,
        .layerCount = 6,
    };

    Image::transitionLayout(oldLayout, newLayout, range, commandBuffer);
}

// ==================== Texture ====================

const vk::raii::ImageView &Texture::getLayerView(const uint32_t layerIndex) const {
    const CubeImage *cubeImage = dynamic_cast<CubeImage *>(&*image);

    if (!cubeImage) {
        throw std::runtime_error("layer-specific views are only supported in cubemap images");
    }

    return cubeImage->getLayerView(layerIndex);
}

const vk::raii::ImageView &Texture::getAttachmentLayerView(const uint32_t layerIndex) const {
    const CubeImage *cubeImage = dynamic_cast<CubeImage *>(&*image);

    if (!cubeImage) {
        throw std::runtime_error("layer-specific views are only supported in cubemap images");
    }

    return cubeImage->getAttachmentLayerView(layerIndex);
}

const vk::raii::ImageView &Texture::getLayerMipView(const uint32_t layerIndex, const uint32_t mipLevel) const {
    const CubeImage *cubeImage = dynamic_cast<CubeImage *>(&*image);

    if (!cubeImage) {
        throw std::runtime_error("layer-specific views are only supported in cubemap images");
    }

    return cubeImage->getLayerMipView(layerIndex, mipLevel);
}

void Texture::generateMipmaps(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                              const vk::raii::Queue &queue, const vk::ImageLayout finalLayout) const {
    const vk::FormatProperties formatProperties = ctx.physicalDevice->getFormatProperties(getFormat());

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    const vk::raii::CommandBuffer commandBuffer = vkutils::cmd::beginSingleTimeCommands(*ctx.device, cmdPool);

    const bool isCubeMap = dynamic_cast<CubeImage *>(&*image) != nullptr;
    const uint32_t layerCount = isCubeMap ? 6 : 1;

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = ***image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = layerCount,
        }
    };

    int32_t mipWidth = image->getExtent().width;
    int32_t mipHeight = image->getExtent().height;

    for (uint32_t i = 1; i < mipLevels; i++) {
        vk::ImageMemoryBarrier currBarrier = barrier;
        currBarrier.subresourceRange.baseMipLevel = i - 1;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
            {},
            nullptr,
            nullptr,
            currBarrier
        );

        const std::array<vk::Offset3D, 2> srcOffsets = {
            {
                {0, 0, 0},
                {mipWidth, mipHeight, 1},
            }
        };

        const std::array<vk::Offset3D, 2> dstOffsets = {
            {
                {0, 0, 0},
                {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1},
            }
        };

        const vk::ImageBlit blit{
            .srcSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i - 1,
                .baseArrayLayer = 0,
                .layerCount = layerCount,
            },
            .srcOffsets = srcOffsets,
            .dstSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i,
                .baseArrayLayer = 0,
                .layerCount = layerCount,
            },
            .dstOffsets = dstOffsets
        };

        commandBuffer.blitImage(
            ***image, vk::ImageLayout::eTransferSrcOptimal,
            ***image, vk::ImageLayout::eTransferDstOptimal,
            blit,
            vk::Filter::eLinear
        );

        vk::ImageMemoryBarrier transBarrier = currBarrier;
        transBarrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        transBarrier.newLayout = finalLayout;
        transBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        transBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            {},
            nullptr,
            nullptr,
            transBarrier
        );

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    vk::ImageMemoryBarrier transBarrier = barrier;
    transBarrier.subresourceRange.baseMipLevel = mipLevels - 1;
    transBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    transBarrier.newLayout = finalLayout;
    transBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    transBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {},
        nullptr,
        nullptr,
        transBarrier
    );

    vkutils::cmd::endSingleTimeCommands(commandBuffer, queue);
}

void Texture::createSampler(const RendererContext &ctx, const vk::SamplerAddressMode addressMode) {
    const vk::PhysicalDeviceProperties properties = ctx.physicalDevice->getProperties();

    const vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = addressMode,
        .addressModeV = addressMode,
        .addressModeW = addressMode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(mipLevels),
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    textureSampler = make_unique<vk::raii::Sampler>(*ctx.device, samplerInfo);
}

// ==================== TextureBuilder ====================

TextureBuilder &TextureBuilder::useFormat(const vk::Format f) {
    format = f;
    return *this;
}

TextureBuilder &TextureBuilder::useLayout(const vk::ImageLayout l) {
    layout = l;
    return *this;
}

TextureBuilder &TextureBuilder::useUsage(const vk::ImageUsageFlags u) {
    usage = u;
    return *this;
}

TextureBuilder &TextureBuilder::asCubemap() {
    isCubemap = true;
    return *this;
}

TextureBuilder &TextureBuilder::asSeparateChannels() {
    isSeparateChannels = true;
    return *this;
}

TextureBuilder &TextureBuilder::asHdr() {
    isHdr = true;
    return *this;
}

TextureBuilder &TextureBuilder::makeMipmaps() {
    hasMipmaps = true;
    return *this;
}

TextureBuilder &TextureBuilder::withSamplerAddressMode(const vk::SamplerAddressMode mode) {
    addressMode = mode;
    return *this;
}

TextureBuilder &TextureBuilder::asUninitialized(const vk::Extent3D extent) {
    isUninitialized = true;
    desiredExtent = extent;
    return *this;
}

TextureBuilder &TextureBuilder::withSwizzle(const std::array<SwizzleComp, 4> sw) {
    swizzle = sw;
    return *this;
}

TextureBuilder &TextureBuilder::fromPaths(const std::vector<std::filesystem::path> &sources) {
    paths = sources;
    return *this;
}

TextureBuilder &TextureBuilder::fromMemory(void *ptr, const vk::Extent3D extent) {
    memorySource = ptr;
    desiredExtent = extent;
    return *this;
}

unique_ptr<Texture> TextureBuilder::create(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                                           const vk::raii::Queue &queue) const {
    checkParams();

    // stupid workaround because std::unique_ptr doesn't have access to the Texture ctor
    unique_ptr<Texture> texture; {
        Texture t;
        texture = make_unique<Texture>(std::move(t));
    }

    const auto [stagingBuffer, extent, layerCount] = isUninitialized
                                                         ? LoadedTextureData{nullptr, *desiredExtent, getLayerCount()}
                                                         : loadFromPaths(ctx);

    texture->mipLevels = hasMipmaps
                             ? static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1
                             : 1;

    const vk::ImageCreateInfo imageInfo{
        .flags = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : static_cast<vk::ImageCreateFlags>(0),
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = extent,
        .mipLevels = texture->mipLevels,
        .arrayLayers = layerCount,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    const bool isDepth = !!(usage & vk::ImageUsageFlagBits::eDepthStencilAttachment);
    const auto aspectFlags = isDepth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    if (isCubemap) {
        texture->image = make_unique<CubeImage>(
            ctx,
            imageInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    } else {
        texture->image = make_unique<Image>(
            ctx,
            imageInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            aspectFlags
        );
    }

    texture->image->createViews(ctx);
    texture->createSampler(ctx, addressMode);

    vkutils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        texture->image->transitionLayout(
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            cmdBuffer
        );

        if (!isUninitialized) {
            texture->image->copyFromBuffer(**stagingBuffer, cmdBuffer);
        }

        if (!hasMipmaps) {
            texture->image->transitionLayout(
                vk::ImageLayout::eTransferDstOptimal,
                layout,
                cmdBuffer
            );
        }
    });

    if (hasMipmaps) {
        texture->generateMipmaps(ctx, cmdPool, queue, layout);
    }

    return texture;
}

void TextureBuilder::checkParams() const {
    if (paths.empty() && !memorySource && !isUninitialized) {
        throw std::runtime_error("no specified data source for texture!");
    }

    if (!paths.empty() && memorySource) {
        throw std::runtime_error("cannot specify two different kinds of texture sources!");
    }

    if (!paths.empty() && isUninitialized) {
        throw std::runtime_error("cannot simultaneously set texture as uninitialized and specify path sources!");
    }

    if (memorySource && isUninitialized) {
        throw std::runtime_error("cannot simultaneously set texture as uninitialized and specify a memory source!");
    }

    if (isCubemap) {
        if (memorySource) {
            throw std::runtime_error("cubemaps from a memory source are currently not supported!");
        }

        if (isSeparateChannels) {
            throw std::runtime_error("cubemaps from separated channels are currently not supported!");
        }

        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            throw std::runtime_error("cubemaps cannot be depth/stencil attachments!");
        }

        if (paths.size() != 6 && !isUninitialized) {
            throw std::runtime_error("invalid layer count for cubemap texture!");
        }
    } else {
        // non-cubemap
        if (isSeparateChannels) {
            if (paths.size() != 3) {
                throw std::runtime_error("unsupported channel count for separate-channelled non-cubemap texture!");
            }
        } else if (!memorySource && paths.size() != 1 && !isUninitialized) {
            throw std::runtime_error("invalid layer count for non-cubemap texture!");
        }
    }

    if (isSeparateChannels) {
        if (isUninitialized) {
            throw std::runtime_error("cannot specify texture as uninitialized and derived from separate channels!");
        }

        if (memorySource) {
            throw std::runtime_error("separate-channeled textures from a memory source are currently not supported!");
        }

        if (vkutils::img::getFormatSizeInBytes(format) != 4) {
            throw std::runtime_error("currently only 4-byte formats are supported when using separate channel mode!");
        }

        if (vkutils::img::getFormatSizeInBytes(format) % 4 != 0) {
            throw std::runtime_error(
                "currently only 4-component formats are supported when using separate channel mode!"
            );
        }

        for (size_t comp = 0; comp < 3; comp++) {
            if (paths[comp].empty()
                && swizzle[comp] != SwizzleComp::ZERO
                && swizzle[comp] != SwizzleComp::ONE
                && swizzle[comp] != SwizzleComp::MAX) {
                throw std::runtime_error("invalid swizzle component for channel provided by an empty path!");
            }
        }
    }
}

uint32_t TextureBuilder::getLayerCount() const {
    if (memorySource) return 1;

    const uint32_t sourcesCount = isUninitialized
        ? (isCubemap ? 6 : 1)
        : paths.size();
    return isSeparateChannels ? sourcesCount / 3 : sourcesCount;
}

TextureBuilder::LoadedTextureData TextureBuilder::loadFromPaths(const RendererContext &ctx) const {
    std::vector<void *> dataSources;
    int texWidth = 0, texHeight = 0, texChannels;

    if (memorySource) {
        dataSources.emplace_back(memorySource);
        texWidth = desiredExtent->width;
        texHeight = desiredExtent->height;
    } else {
        bool isFirstNonEmpty = true;

        for (const auto & path: paths) {
            if (path.empty()) {
                dataSources.push_back(nullptr);
                continue;
            }

            stbi_set_flip_vertically_on_load(isHdr);
            const int desiredChannels = isSeparateChannels ? STBI_grey : STBI_rgb_alpha;
            void *src;

            int currTexWidth, currTexHeight;

            if (isHdr) {
                src = stbi_loadf(path.string().c_str(), &currTexWidth, &currTexHeight, &texChannels, desiredChannels);
            } else {
                src = stbi_load(path.string().c_str(), &currTexWidth, &currTexHeight, &texChannels, desiredChannels);
            }

            if (!src) {
                throw std::runtime_error("failed to load texture image!");
            }

            if (isFirstNonEmpty) {
                texWidth = currTexWidth;
                texHeight = currTexHeight;
                isFirstNonEmpty = false;

            } else if (texWidth != currTexWidth || texHeight != currTexHeight) {
                throw std::runtime_error("size mismatch while loading a texture from paths!");
            }

            dataSources.push_back(src);
        }
    }

    const uint32_t layerCount = getLayerCount();
    const vk::DeviceSize formatSize = vkutils::img::getFormatSizeInBytes(format);
    const vk::DeviceSize layerSize = texWidth * texHeight * formatSize;
    const vk::DeviceSize textureSize = layerSize * layerCount;

    constexpr uint32_t componentCount = 4;
    if (formatSize % componentCount != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    if (isSeparateChannels) {
        dataSources = {mergeChannels(dataSources, textureSize, componentCount)};
    }

    for (const auto &source: dataSources) {
        performSwizzle(static_cast<uint8_t *>(source), layerSize);
    }

    auto stagingBuffer = make_unique<Buffer>(
        **ctx.allocator,
        textureSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void *data = stagingBuffer->map();

    for (size_t i = 0; i < getLayerCount(); i++) {
        const size_t offset = layerSize * i;
        memcpy(static_cast<char *>(data) + offset, dataSources[i], static_cast<size_t>(layerSize));

        if (isSeparateChannels) {
            free(dataSources[i]);
        } else if (!memorySource) {
            stbi_image_free(dataSources[i]);
        }
    }

    stagingBuffer->unmap();

    return {
        .stagingBuffer = std::move(stagingBuffer),
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1u
        },
        .layerCount = layerCount
    };
}

void *TextureBuilder::mergeChannels(const std::vector<void *> &channelsData, const size_t textureSize,
                                    const size_t componentCount) {
    auto *merged = static_cast<uint8_t *>(malloc(textureSize));
    if (!merged) {
        throw std::runtime_error("malloc failed");
    }

    for (size_t i = 0; i < textureSize; i++) {
        if (i % componentCount == componentCount - 1 || !channelsData[i % componentCount]) {
            merged[i] = 0; // todo - utilize alpha
        } else {
            merged[i] = static_cast<uint8_t *>(channelsData[i % componentCount])[i / componentCount];
        }
    }

    return merged;
}

void TextureBuilder::performSwizzle(uint8_t *data, const size_t size) const {
    constexpr size_t componentCount = 4;

    for (size_t i = 0; i < size / componentCount; i++) {
        const uint8_t r = data[componentCount * i];
        const uint8_t g = data[componentCount * i + 1];
        const uint8_t b = data[componentCount * i + 2];
        const uint8_t a = data[componentCount * i + 3];

        for (size_t comp = 0; comp < componentCount; comp++) {
            switch (swizzle[comp]) {
                case SwizzleComp::R:
                    data[componentCount * i + comp] = r;
                    break;
                case SwizzleComp::G:
                    data[componentCount * i + comp] = g;
                    break;
                case SwizzleComp::B:
                    data[componentCount * i + comp] = b;
                    break;
                case SwizzleComp::A:
                    data[componentCount * i + comp] = a;
                    break;
                case SwizzleComp::ZERO:
                    data[componentCount * i + comp] = 0;
                    break;
                case SwizzleComp::ONE:
                    data[componentCount * i + comp] = 1;
                    break;
                case SwizzleComp::MAX:
                    data[componentCount * i + comp] = std::numeric_limits<uint8_t>::max();
                    break;
            }
        }
    }
}

// ==================== utils ====================

unique_ptr<vk::raii::ImageView>
vkutils::img::createImageView(const RendererContext &ctx, const vk::Image image, const vk::Format format,
                            const vk::ImageAspectFlags aspectFlags, const uint32_t baseMipLevel,
                            const uint32_t mipLevels, const uint32_t layer) {
    const vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = baseMipLevel,
            .levelCount = mipLevels,
            .baseArrayLayer = layer,
            .layerCount = 1,
        }
    };

    return make_unique<vk::raii::ImageView>(*ctx.device, createInfo);
}

unique_ptr<vk::raii::ImageView>
vkutils::img::createCubeImageView(const RendererContext &ctx, const vk::Image image, const vk::Format format,
                                const vk::ImageAspectFlags aspectFlags, const uint32_t baseMipLevel,
                                const uint32_t mipLevels) {
    const vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::eCube,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = baseMipLevel,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6,
        }
    };

    return make_unique<vk::raii::ImageView>(*ctx.device, createInfo);
}

size_t vkutils::img::getFormatSizeInBytes(const vk::Format format) {
    switch (format) {
        case vk::Format::eB8G8R8A8Srgb:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eR8G8B8A8Unorm:
            return 4;
        case vk::Format::eR16G16B16Sfloat:
            return 6;
        case vk::Format::eR16G16B16A16Sfloat:
            return 8;
        case vk::Format::eR32G32B32Sfloat:
            return 12;
        case vk::Format::eR32G32B32A32Sfloat:
            return 16;
        default:
            throw std::runtime_error("unexpected format in utils::img::getFormatSizeInBytes");
    }
}
