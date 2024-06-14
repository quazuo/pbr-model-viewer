#include "image.h"

#include <filesystem>

#include "deps/stb/stb_image.h"

#include "buffer.h"
#include "cmd.h"
#include "src/render/renderer.h"

Image::Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo, const vk::MemoryPropertyFlags properties)
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

// ==================== CubeImage ====================

CubeImage::CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                     const vk::MemoryPropertyFlags properties) : Image(ctx, imageInfo, properties) {
}

void CubeImage::createView(const RendererContext &ctx, const vk::Format format, const vk::ImageAspectFlags aspectFlags,
                           const std::uint32_t mipLevels) {
    view = utils::img::createCubeImageView(ctx, **image, format, aspectFlags, mipLevels);
}

void CubeImage::copyFromBuffer(const RendererContext &ctx, const vk::Buffer buffer,
                               const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue) {
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

    utils::cmd::doSingleTimeCommands(*ctx.device, cmdPool, queue, [&](const auto &cmdBuffer) {
        cmdBuffer.copyBufferToImage(
            buffer,
            **image,
            vk::ImageLayout::eTransferDstOptimal,
            region
        );
    });
}

// ==================== Texture ====================

void Texture::generateMipmaps(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                              const vk::raii::Queue &queue, const vk::ImageLayout finalLayout) const {
    const vk::FormatProperties formatProperties = ctx.physicalDevice->getFormatProperties(format);

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    const vk::raii::CommandBuffer commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, cmdPool);

    const bool isCubeMap = dynamic_cast<CubeImage *>(&*image) != nullptr;
    const std::uint32_t layerCount = isCubeMap ? 6 : 1;

    const vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = *image->get(),
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
            *image->get(), vk::ImageLayout::eTransferSrcOptimal,
            *image->get(), vk::ImageLayout::eTransferDstOptimal,
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

    utils::cmd::endSingleTimeCommands(commandBuffer, queue);
}

void Texture::createSampler(const RendererContext &ctx) {
    const vk::PhysicalDeviceProperties properties = ctx.physicalDevice->getProperties();

    const bool isFormatUint = format == vk::Format::eR8Uint || format == vk::Format::eR8G8B8A8Uint;
    const vk::Filter filter = isFormatUint
                                  ? vk::Filter::eNearest
                                  : vk::Filter::eLinear;
    const vk::SamplerMipmapMode mipmapMode = isFormatUint
                                                 ? vk::SamplerMipmapMode::eNearest
                                                 : vk::SamplerMipmapMode::eLinear;

    const vk::SamplerCreateInfo samplerInfo{
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = mipmapMode,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
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

    textureSampler = std::make_unique<vk::raii::Sampler>(*ctx.device, samplerInfo);
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

TextureBuilder &TextureBuilder::makeMipmaps() {
    hasMipmaps = true;
    return *this;
}

TextureBuilder &TextureBuilder::fromPaths(const std::vector<std::filesystem::path> &paths) {
    layerCount = paths.size();
    sources = paths;
    return *this;
}

TextureBuilder &TextureBuilder::fromDataPtr(vk::Extent3D extent, void *data, const size_t layers) {
    layerCount = layers;
    sources = std::make_pair(extent, data);
    return *this;
}

Texture TextureBuilder::create(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                               const vk::raii::Queue &queue) const {
    checkParams();

    Texture texture;

    texture.format = format;

    std::vector<void *> dataSources;
    int texWidth, texHeight, texChannels;

    if (std::holds_alternative<path_vec_t>(sources)) {
        const auto &paths = std::get<path_vec_t>(sources);

        for (const auto &path: paths) {
            void *src = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            if (!src) {
                throw std::runtime_error("failed to load texture image!");
            }

            dataSources.push_back(src);
        }
    } else if (std::holds_alternative<ptr_source_t>(sources)) {
        const auto &[extent, ptr] = std::get<ptr_source_t>(sources);
        texWidth = extent.width;
        texHeight = extent.height;
        dataSources.push_back(ptr);
    }

    texture.mipLevels = hasMipmaps
                            ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1
                            : 1;
    const vk::DeviceSize layerSize = texWidth * texHeight * utils::img::getFormatSizeInBytes(format);
    const vk::DeviceSize textureSize = layerSize * layerCount;

    Buffer stagingBuffer{
        ctx.allocator->get(),
        textureSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();

    if (std::holds_alternative<path_vec_t>(sources)) {
        for (size_t i = 0; i < dataSources.size(); i++) {
            const size_t offset = layerSize * i;
            memcpy(static_cast<char *>(data) + offset, dataSources[i], static_cast<size_t>(layerSize));
            stbi_image_free(dataSources[i]);
        }
    } else if (std::holds_alternative<ptr_source_t>(sources)) {
        memcpy(data, dataSources[0], static_cast<size_t>(textureSize));
    }

    stagingBuffer.unmap();

    const vk::ImageCreateInfo imageInfo{
        .flags = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : static_cast<vk::ImageCreateFlags>(0),
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1,
        },
        .mipLevels = texture.mipLevels,
        .arrayLayers = layerCount,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    if (isCubemap) {
        texture.image = std::make_unique<CubeImage>(
            ctx,
            imageInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    } else {
        texture.image = std::make_unique<Image>(
            ctx,
            imageInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    }

    utils::img::transitionImageLayout(
        ctx,
        *texture.image->get(),
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        texture.mipLevels,
        layerCount,
        cmdPool,
        queue
    );

    texture.image->copyFromBuffer(ctx, stagingBuffer.get(), cmdPool, queue);

    texture.image->createView(
        ctx,
        format,
        vk::ImageAspectFlagBits::eColor,
        texture.mipLevels
    );

    texture.createSampler(ctx);

    if (hasMipmaps) {
        texture.generateMipmaps(ctx, cmdPool, queue, layout);
    } else {
        utils::img::transitionImageLayout(
            ctx,
            *texture.image->get(),
            vk::ImageLayout::eTransferDstOptimal,
            layout,
            texture.mipLevels,
            layerCount,
            cmdPool,
            queue
        );
    }

    return texture;
}

void TextureBuilder::checkParams() const {
    if (std::holds_alternative<std::nullopt_t>(sources)) {
        throw std::runtime_error("no specified data source for texture!");
    }

    if (isCubemap && layerCount != 6) {
        throw std::runtime_error("invalid layer count for cubemap texture!");
    }

    if (!isCubemap && layerCount != 1) {
        throw std::runtime_error("invalid layer count for non-cubemap texture!");
    }
}

// ==================== utils ====================

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

std::unique_ptr<vk::raii::ImageView>
utils::img::createCubeImageView(const RendererContext &ctx, const vk::Image image, const vk::Format format,
                                const vk::ImageAspectFlags aspectFlags, const std::uint32_t mipLevels) {
    const vk::ImageViewCreateInfo createInfo{
        .image = image,
        .viewType = vk::ImageViewType::eCube,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6,
        }
    };

    return std::make_unique<vk::raii::ImageView>(*ctx.device, createInfo);
}

void utils::img::transitionImageLayout(const RendererContext &ctx, const vk::Image image,
                                       const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout,
                                       const std::uint32_t mipLevels, const std::uint32_t layerCount,
                                       const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue) {
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
            .layerCount = layerCount,
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

size_t utils::img::getFormatSizeInBytes(const vk::Format format) {
    switch (format) {
        case vk::Format::eB8G8R8A8Srgb:
        case vk::Format::eR8G8B8A8Srgb:
        case vk::Format::eR8G8B8A8Uint:
            return 4;
        case vk::Format::eR8Uint:
            return 1;
        default:
            throw std::runtime_error("unexpected format in utils::img::getFormatSizeInBytes");
    }
}
