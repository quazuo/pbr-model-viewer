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

shared_ptr<vk::raii::ImageView> Image::getView(const RendererContext &ctx) {
    return getCachedView(ctx, {0, mipLevels, 0, 1});
}

shared_ptr<vk::raii::ImageView> Image::getMipView(const RendererContext &ctx, const uint32_t mipLevel) {
    return getCachedView(ctx, {mipLevel, 1, 0, 1});
}

shared_ptr<vk::raii::ImageView> Image::getLayerView(const RendererContext &ctx, const uint32_t layer) {
    return getCachedView(ctx, {0, mipLevels, layer, 1});
}

shared_ptr<vk::raii::ImageView> Image::getLayerMipView(const RendererContext &ctx, const uint32_t layer,
                                                       const uint32_t mipLevel) {
    return getCachedView(ctx, {mipLevel, 1, layer, 1});
}

shared_ptr<vk::raii::ImageView> Image::getCachedView(const RendererContext &ctx, ViewParams params) {
    if (cachedViews.contains(params)) {
        return cachedViews.at(params);
    }

    const auto &[baseMip, mipCount, baseLayer, layerCount] = params;

    auto view = layerCount == 1
                    ? vkutils::img::createImageView(ctx, **image, format, aspectMask, baseMip, mipCount, baseLayer)
                    : vkutils::img::createCubeImageView(ctx, **image, format, aspectMask, baseMip, mipCount);
    auto viewPtr = make_shared<vk::raii::ImageView>(std::move(view));
    cachedViews.emplace(params, viewPtr);
    return viewPtr;
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

    const auto &[srcAccessMask, dstAccessMask, srcStage, dstStage] =
            transitionBarrierSchemes.at({oldLayout, newLayout});

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

void Image::saveToFile(const RendererContext &ctx, const std::filesystem::path &path) const {
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

    vkutils::cmd::doSingleTimeCommands(ctx, [&](const auto &cmdBuffer) {
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

    vkutils::cmd::doSingleTimeCommands(ctx, [&](const auto &commandBuffer) {
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

    vkutils::cmd::doSingleTimeCommands(ctx, [&](const auto &cmdBuffer) {
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

shared_ptr<vk::raii::ImageView> CubeImage::getView(const RendererContext &ctx) {
    return getCachedView(ctx, {0, 1, 0, 6});
}

shared_ptr<vk::raii::ImageView> CubeImage::getMipView(const RendererContext &ctx, const uint32_t mipLevel) {
    return getCachedView(ctx, {mipLevel, 1, 0, 6});
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

void Texture::generateMipmaps(const RendererContext &ctx, const vk::ImageLayout finalLayout) const {
    const vk::FormatProperties formatProperties = ctx.physicalDevice->getFormatProperties(getFormat());

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    const vk::raii::CommandBuffer commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

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

    for (uint32_t i = 1; i < image->getMipLevels(); i++) {
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
    transBarrier.subresourceRange.baseMipLevel = image->getMipLevels() - 1;
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

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);
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
        .maxLod = static_cast<float>(image->getMipLevels()),
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

TextureBuilder &TextureBuilder::withSwizzle(const std::array<SwizzleComponent, 4> sw) {
    swizzle = sw;
    return *this;
}

TextureBuilder &TextureBuilder::fromPaths(const std::vector<std::filesystem::path> &sources) {
    paths = sources;
    return *this;
}

TextureBuilder &TextureBuilder::fromMemory(void *ptr, const vk::Extent3D extent) {
    if (!ptr) {
        throw std::invalid_argument("cannot specify null memory source!");
    }

    memorySource = ptr;
    desiredExtent = extent;
    return *this;
}

TextureBuilder &TextureBuilder::fromSwizzleFill(vk::Extent3D extent) {
    isFromSwizzleFill = true;
    desiredExtent = extent;
    return *this;
}

unique_ptr<Texture> TextureBuilder::create(const RendererContext &ctx) const {
    checkParams();

    // stupid workaround because std::unique_ptr doesn't have access to the Texture ctor
    unique_ptr<Texture> texture; {
        Texture t;
        texture = make_unique<Texture>(std::move(t));
    }

    LoadedTextureData loadedTexData;

    if (isUninitialized) loadedTexData = {{}, *desiredExtent, getLayerCount()};
    else if (!paths.empty()) loadedTexData = loadFromPaths();
    else if (memorySource) loadedTexData = loadFromMemory();
    else if (isFromSwizzleFill) loadedTexData = loadFromSwizzleFill();

    const auto extent = loadedTexData.extent;
    const auto stagingBuffer = isUninitialized ? nullptr : makeStagingBuffer(ctx, loadedTexData);

    uint32_t mipLevels = 1;
    if (hasMipmaps) {
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(extent.width, extent.height)))) + 1;
    }

    const vk::ImageCreateInfo imageInfo{
        .flags = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : static_cast<vk::ImageCreateFlags>(0),
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = extent,
        .mipLevels = mipLevels,
        .arrayLayers = loadedTexData.layerCount,
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

    texture->createSampler(ctx, addressMode);

    vkutils::cmd::doSingleTimeCommands(ctx, [&](const auto &cmdBuffer) {
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
        texture->generateMipmaps(ctx, layout);
    }

    return texture;
}

void TextureBuilder::checkParams() const {
    if (paths.empty() && !memorySource && !isFromSwizzleFill && !isUninitialized) {
        throw std::invalid_argument("no specified data source for texture!");
    }

    size_t sourcesCount = 0;
    if (!paths.empty()) sourcesCount++;
    if (memorySource) sourcesCount++;
    if (isFromSwizzleFill) sourcesCount++;

    if (sourcesCount > 1) {
        throw std::invalid_argument("cannot specify more than one texture source!");
    }

    if (sourcesCount != 0 && isUninitialized) {
        throw std::invalid_argument("cannot simultaneously set texture as uninitialized and specify sources!");
    }

    if (isCubemap) {
        if (memorySource) {
            throw std::invalid_argument("cubemaps from a memory source are currently not supported!");
        }

        if (isSeparateChannels) {
            throw std::invalid_argument("cubemaps from separated channels are currently not supported!");
        }

        if (isFromSwizzleFill) {
            throw std::invalid_argument("cubemaps from swizzle fill are currently not supported!");
        }

        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            throw std::invalid_argument("cubemaps cannot be depth/stencil attachments!");
        }

        if (paths.size() != 6 && !isUninitialized) {
            throw std::invalid_argument("invalid layer count for cubemap texture!");
        }
    } else {
        // non-cubemap
        if (isSeparateChannels) {
            if (paths.size() != 3) {
                throw std::invalid_argument("unsupported channel count for separate-channelled non-cubemap texture!");
            }
        } else if (!memorySource && !isFromSwizzleFill && !isUninitialized && paths.size() != 1) {
            throw std::invalid_argument("invalid layer count for non-cubemap texture!");
        }
    }

    if (isSeparateChannels) {
        if (paths.empty()) {
            throw std::invalid_argument("separate-channeled textures must provide path sources!");
        }

        if (vkutils::img::getFormatSizeInBytes(format) != 4) {
            throw std::invalid_argument(
                "currently only 4-byte formats are supported when using separate channel mode!");
        }

        if (vkutils::img::getFormatSizeInBytes(format) % 4 != 0) {
            throw std::invalid_argument(
                "currently only 4-component formats are supported when using separate channel mode!"
            );
        }

        if (swizzle) {
            for (size_t comp = 0; comp < 3; comp++) {
                if (paths[comp].empty()
                    && (*swizzle)[comp] != SwizzleComponent::ZERO
                    && (*swizzle)[comp] != SwizzleComponent::ONE
                    && (*swizzle)[comp] != SwizzleComponent::MAX
                    && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                    throw std::invalid_argument("invalid swizzle component for channel provided by an empty path!");
                }
            }
        }
    }

    if (isFromSwizzleFill) {
        if (!swizzle) {
            throw std::invalid_argument("textures filled from swizzle must provide a swizzle!");
        }

        for (size_t comp = 0; comp < 3; comp++) {
            if ((*swizzle)[comp] != SwizzleComponent::ZERO
                && (*swizzle)[comp] != SwizzleComponent::ONE
                && (*swizzle)[comp] != SwizzleComponent::MAX
                && (*swizzle)[comp] != SwizzleComponent::HALF_MAX) {
                throw std::invalid_argument("invalid swizzle component for swizzle-filled texture!");
            }
        }
    }
}

uint32_t TextureBuilder::getLayerCount() const {
    if (memorySource || isFromSwizzleFill) return 1;

    const uint32_t sourcesCount = isUninitialized
                                      ? (isCubemap ? 6 : 1)
                                      : paths.size();
    return isSeparateChannels ? sourcesCount / 3 : sourcesCount;
}

TextureBuilder::LoadedTextureData TextureBuilder::loadFromPaths() const {
    std::vector<void *> dataSources;
    int texWidth = 0, texHeight = 0, texChannels;
    bool isFirstNonEmpty = true;

    for (const auto &path: paths) {
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
            throw std::runtime_error("failed to load texture image at path: " + path.string());
        }

        if (isFirstNonEmpty && !desiredExtent) {
            texWidth = currTexWidth;
            texHeight = currTexHeight;
            isFirstNonEmpty = false;
        } else if (texWidth != currTexWidth || texHeight != currTexHeight) {
            throw std::runtime_error("size mismatch while loading a texture from paths!");
        }

        dataSources.push_back(src);
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

    if (swizzle) {
        for (const auto &source: dataSources) {
            performSwizzle(static_cast<uint8_t *>(source), layerSize);
        }
    }

    return {
        .sources = dataSources,
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1u
        },
        .layerCount = layerCount
    };
}

TextureBuilder::LoadedTextureData TextureBuilder::loadFromMemory() const {
    const std::vector<void *> dataSources = {memorySource};

    const uint32_t texWidth = desiredExtent->width;
    const uint32_t texHeight = desiredExtent->height;

    const uint32_t layerCount = getLayerCount();
    const vk::DeviceSize formatSize = vkutils::img::getFormatSizeInBytes(format);
    const vk::DeviceSize layerSize = texWidth * texHeight * formatSize;

    constexpr uint32_t componentCount = 4;
    if (formatSize % componentCount != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    if (swizzle) {
        for (const auto &source: dataSources) {
            performSwizzle(static_cast<uint8_t *>(source), layerSize);
        }
    }

    return {
        .sources = dataSources,
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1u
        },
        .layerCount = layerCount
    };
}

TextureBuilder::LoadedTextureData TextureBuilder::loadFromSwizzleFill() const {
    const uint32_t texWidth = desiredExtent->width;
    const uint32_t texHeight = desiredExtent->height;
    const uint32_t layerCount = getLayerCount();
    const vk::DeviceSize formatSize = vkutils::img::getFormatSizeInBytes(format);
    const vk::DeviceSize layerSize = texWidth * texHeight * formatSize;
    const vk::DeviceSize textureSize = layerSize * layerCount;

    constexpr uint32_t componentCount = 4;
    if (formatSize % componentCount != 0) {
        throw std::runtime_error("texture formats with component count other than 4 are currently unsupported!");
    }

    const std::vector<void *> dataSources = {malloc(textureSize)};

    for (const auto &source: dataSources) {
        performSwizzle(static_cast<uint8_t *>(source), layerSize);
    }

    return {
        .sources = dataSources,
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1u
        },
        .layerCount = layerCount
    };
}

unique_ptr<Buffer> TextureBuilder::makeStagingBuffer(const RendererContext &ctx, const LoadedTextureData &data) const {
    const uint32_t layerCount = getLayerCount();
    const vk::DeviceSize formatSize = vkutils::img::getFormatSizeInBytes(format);
    const vk::DeviceSize layerSize = data.extent.width * data.extent.height * formatSize;
    const vk::DeviceSize textureSize = layerSize * layerCount;

    auto stagingBuffer = make_unique<Buffer>(
        **ctx.allocator,
        textureSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void *mapped = stagingBuffer->map();

    for (size_t i = 0; i < getLayerCount(); i++) {
        const size_t offset = layerSize * i;
        memcpy(static_cast<char *>(mapped) + offset, data.sources[i], static_cast<size_t>(layerSize));

        if (isSeparateChannels || isFromSwizzleFill) {
            free(data.sources[i]);
        } else if (!memorySource) {
            stbi_image_free(data.sources[i]);
        }
    }

    stagingBuffer->unmap();

    return stagingBuffer;
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
    if (!swizzle) {
        throw std::runtime_error("unexpected empty swizzle optional in TextureBuilder::performSwizzle");
    }

    constexpr size_t componentCount = 4;

    for (size_t i = 0; i < size / componentCount; i++) {
        const uint8_t r = data[componentCount * i];
        const uint8_t g = data[componentCount * i + 1];
        const uint8_t b = data[componentCount * i + 2];
        const uint8_t a = data[componentCount * i + 3];

        for (size_t comp = 0; comp < componentCount; comp++) {
            switch ((*swizzle)[comp]) {
                case SwizzleComponent::R:
                    data[componentCount * i + comp] = r;
                    break;
                case SwizzleComponent::G:
                    data[componentCount * i + comp] = g;
                    break;
                case SwizzleComponent::B:
                    data[componentCount * i + comp] = b;
                    break;
                case SwizzleComponent::A:
                    data[componentCount * i + comp] = a;
                    break;
                case SwizzleComponent::ZERO:
                    data[componentCount * i + comp] = 0;
                    break;
                case SwizzleComponent::ONE:
                    data[componentCount * i + comp] = 1;
                    break;
                case SwizzleComponent::MAX:
                    data[componentCount * i + comp] = std::numeric_limits<uint8_t>::max();
                    break;
                case SwizzleComponent::HALF_MAX:
                    data[componentCount * i + comp] = std::numeric_limits<uint8_t>::max() / 2;
                    break;
            }
        }
    }
}

// ==================== RenderTarget ====================

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, const vk::Format format)
    : view(std::move(view)), format(format) {
}

RenderTarget::RenderTarget(shared_ptr<vk::raii::ImageView> view, shared_ptr<vk::raii::ImageView> resolveView,
                           const vk::Format format)
    : view(std::move(view)), resolveView(std::move(resolveView)), format(format) {
}

RenderTarget::RenderTarget(const RendererContext &ctx, const Texture &texture)
    : view(texture.getImage().getView(ctx)), format(texture.getFormat()) {
}

vk::RenderingAttachmentInfo RenderTarget::getAttachmentInfo() const {
    const auto layout = vkutils::img::isDepthFormat(format)
                            ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                            : vk::ImageLayout::eColorAttachmentOptimal;

    vk::ClearValue clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    if (vkutils::img::isDepthFormat(format)) {
        clearValue = vk::ClearDepthStencilValue{
            .depth = 1.0f,
            .stencil = 0,
        };
    }

    vk::RenderingAttachmentInfo info{
        .imageView = **view,
        .imageLayout = layout,
        .loadOp = loadOp,
        .storeOp = storeOp,
        .clearValue = clearValue,
    };

    if (resolveView) {
        info.resolveMode = vk::ResolveModeFlagBits::eAverage;
        info.resolveImageView = **resolveView;
        info.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    return info;
}

void RenderTarget::overrideAttachmentConfig(const vk::AttachmentLoadOp loadOp, const vk::AttachmentStoreOp storeOp) {
    this->loadOp = loadOp;
    this->storeOp = storeOp;
}

// ==================== utils ====================

vk::raii::ImageView
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
        },
    };

    return {*ctx.device, createInfo};
}

vk::raii::ImageView
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

    return {*ctx.device, createInfo};
}

bool vkutils::img::isDepthFormat(const vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eD32Sfloat:
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return true;
        default:
            return false;
    }
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
