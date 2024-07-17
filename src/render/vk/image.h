#pragma once

#include <filesystem>
#include <map>

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"
#include "src/render/globals.h"

class Buffer;

struct RendererContext;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and are mostly suited for swap chain related logic.
 */
class Image {
protected:
    VmaAllocator allocator{};
    unique_ptr<VmaAllocation> allocation{};
    unique_ptr<vk::raii::Image> image;
    unique_ptr<vk::raii::ImageView> view;
    unique_ptr<vk::raii::ImageView> attachmentView;
    std::vector<unique_ptr<vk::raii::ImageView> > mipViews;
    vk::Extent3D extent;
    vk::Format format{};
    uint32_t mipLevels;
    vk::ImageAspectFlags aspectMask;

public:
    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                   vk::MemoryPropertyFlags properties, vk::ImageAspectFlags aspect);

    virtual ~Image();

    Image(const Image &other) = delete;

    Image(Image &&other) = delete;

    Image &operator=(const Image &other) = delete;

    Image &operator=(Image &&other) = delete;

    /**
     * Returns a raw handle to the actual Vulkan image.
     * @return Handle to the image.
     */
    [[nodiscard]] const vk::raii::Image &operator*() const { return *image; }

    /**
     * Returns a raw handle to the actual Vulkan image view associated with this image.
     * @return Handle to the image view.
     */
    [[nodiscard]] const vk::raii::ImageView &getView() const { return *view; }

    [[nodiscard]] const vk::raii::ImageView &getAttachmentView() const { return *attachmentView; }

    [[nodiscard]] const vk::raii::ImageView &getMipView(const uint32_t mipLevel) const { return *mipViews[mipLevel]; }

    [[nodiscard]] vk::Extent3D getExtent() const { return extent; }

    [[nodiscard]] vk::Extent2D getExtent2d() const { return {extent.width, extent.height}; }

    [[nodiscard]] vk::Format getFormat() const { return format; }

    virtual void createViews(const RendererContext &ctx);

    /**
     * Records commands that copy the contents of a given buffer to this image.
     *
     * @param buffer Buffer from which to copy.
     * @param commandBuffer Command buffer to which the commands will be recorded.
     */
    virtual void copyFromBuffer(vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer);

    virtual void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                  const vk::raii::CommandBuffer &commandBuffer) const;

    void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                          vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &commandBuffer) const;

    void saveToFile(const RendererContext &ctx, const std::filesystem::path &path,
                    const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue) const;
};

class CubeImage final : public Image {
    std::vector<unique_ptr<vk::raii::ImageView> > layerViews;
    std::vector<unique_ptr<vk::raii::ImageView> > attachmentLayerViews;
    std::vector<std::vector<unique_ptr<vk::raii::ImageView> > > layerMipViews; // layerMipViews[layer][mip]

public:
    explicit CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                       vk::MemoryPropertyFlags properties);

    /**
     * Returns a raw handle to the actual Vulkan image view associated with a specific layer of this image.
     * @return Handle to the layer's image view.
     *
     * @param layerIndex Layer index to which the view should refer.
     */
    [[nodiscard]] const vk::raii::ImageView &
    getLayerView(const uint32_t layerIndex) const { return *layerViews[layerIndex]; }

    [[nodiscard]] const vk::raii::ImageView &
    getAttachmentLayerView(const uint32_t layerIndex) const { return *attachmentLayerViews[layerIndex]; }

    [[nodiscard]] const vk::raii::ImageView &
    getLayerMipView(const uint32_t layerIndex, const uint32_t mipLevel) const {
        return *layerMipViews[layerIndex][mipLevel];
    }

    void createViews(const RendererContext &ctx) override;

    void copyFromBuffer(vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer) override;

    void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                          const vk::raii::CommandBuffer &commandBuffer) const override;
};

class Texture {
    unique_ptr<Image> image;
    unique_ptr<vk::raii::Sampler> textureSampler;
    uint32_t mipLevels{};

    friend class TextureBuilder;

    Texture() = default;

public:
    [[nodiscard]] const Image &getImage() const { return *image; }

    [[nodiscard]] const vk::raii::Sampler &getSampler() const { return *textureSampler; }

    [[nodiscard]] uint32_t getMipLevels() const { return mipLevels; }

    [[nodiscard]] vk::Format getFormat() const { return image->getFormat(); }

    [[nodiscard]] const vk::raii::ImageView &getView() const { return image->getView(); }

    [[nodiscard]] const vk::raii::ImageView &getAttachmentView() const { return image->getAttachmentView(); }

    [[nodiscard]] const vk::raii::ImageView &getMipView(const uint32_t mipLevel) const {
        return image->getMipView(mipLevel);
    }

    [[nodiscard]] const vk::raii::ImageView &getLayerView(uint32_t layerIndex) const;

    [[nodiscard]] const vk::raii::ImageView &getAttachmentLayerView(uint32_t layerIndex) const;

    [[nodiscard]] const vk::raii::ImageView &getLayerMipView(uint32_t layerIndex, uint32_t mipLevel) const;

    void generateMipmaps(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                         const vk::raii::Queue &queue, vk::ImageLayout finalLayout) const;

private:
    void createSampler(const RendererContext &ctx, vk::SamplerAddressMode addressMode);
};

enum class SwizzleComp {
    R,
    G,
    B,
    A,
    ZERO,
    ONE,
    MAX
};

class TextureBuilder {
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc
                                | vk::ImageUsageFlagBits::eTransferDst
                                | vk::ImageUsageFlagBits::eSampled;
    bool isCubemap = false;
    bool isSeparateChannels = false;
    bool isHdr = false;
    bool hasMipmaps = false;
    bool isUninitialized = false;

    std::array<SwizzleComp, 4> swizzle{SwizzleComp::R, SwizzleComp::G, SwizzleComp::B, SwizzleComp::A};

    vk::SamplerAddressMode addressMode = vk::SamplerAddressMode::eClampToEdge;

    std::optional<vk::Extent3D> desiredExtent;

    std::vector<std::filesystem::path> paths;
    void *memorySource = nullptr;

    struct LoadedTextureData {
        unique_ptr<Buffer> stagingBuffer;
        vk::Extent3D extent;
        uint32_t layerCount;
    };

public:
    TextureBuilder &useFormat(vk::Format f);

    TextureBuilder &useLayout(vk::ImageLayout l);

    TextureBuilder &useUsage(vk::ImageUsageFlags u);

    TextureBuilder &asCubemap();

    TextureBuilder &asSeparateChannels();

    TextureBuilder &asHdr();

    TextureBuilder &makeMipmaps();

    TextureBuilder &withSamplerAddressMode(vk::SamplerAddressMode mode);

    TextureBuilder &asUninitialized(vk::Extent3D extent);

    TextureBuilder &withSwizzle(std::array<SwizzleComp, 4> sw);

    TextureBuilder &fromPaths(const std::vector<std::filesystem::path> &sources);

    TextureBuilder &fromMemory(void *ptr, vk::Extent3D extent);

    [[nodiscard]] unique_ptr<Texture>
    create(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue) const;

private:
    void checkParams() const;

    [[nodiscard]] uint32_t getLayerCount() const;

    [[nodiscard]] LoadedTextureData loadFromPaths(const RendererContext &ctx) const;

    static void *mergeChannels(const std::vector<void *> &channelsData, size_t textureSize, size_t componentCount);

    void performSwizzle(uint8_t *data, size_t size) const;
};

namespace utils::img {
    [[nodiscard]] unique_ptr<vk::raii::ImageView>
    createImageView(const RendererContext &ctx, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags,
                    uint32_t baseMipLevel, uint32_t mipLevels, uint32_t layer);

    [[nodiscard]] unique_ptr<vk::raii::ImageView>
    createCubeImageView(const RendererContext &ctx, vk::Image image, vk::Format format,
                        vk::ImageAspectFlags aspectFlags, uint32_t baseMipLevel, uint32_t mipLevels);

    void saveToFile(...);

    [[nodiscard]] size_t getFormatSizeInBytes(vk::Format format);
}

struct ImageBarrierInfo {
    vk::AccessFlagBits srcAccessMask;
    vk::AccessFlagBits dstAccessMask;
    vk::PipelineStageFlagBits srcStage;
    vk::PipelineStageFlagBits dstStage;
};

static std::map<std::pair<vk::ImageLayout, vk::ImageLayout>, ImageBarrierInfo> transitionBarrierSchemes{
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal},
        {
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .srcStage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dstStage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal},
        {
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .srcStage = vk::PipelineStageFlagBits::eTopOfPipe,
            .dstStage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .srcStage = vk::PipelineStageFlagBits::eTransfer,
            .dstStage = vk::PipelineStageFlagBits::eFragmentShader,
        }
    },
    {
        {vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .srcStage = vk::PipelineStageFlagBits::eTransfer,
            .dstStage = vk::PipelineStageFlagBits::eFragmentShader,
        }
    },
    {
        {vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal},
        {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .srcStage = vk::PipelineStageFlagBits::eFragmentShader,
            .dstStage = vk::PipelineStageFlagBits::eTransfer,
        }
    },
    {
        {vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal},
        {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .srcStage = vk::PipelineStageFlagBits::eFragmentShader,
            .dstStage = vk::PipelineStageFlagBits::eTransfer,
        }
    }
};
