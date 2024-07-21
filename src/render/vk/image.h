#pragma once

#include <filesystem>
#include <map>

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"
#include "src/render/globals.h"

class Buffer;

struct RendererContext;

struct ViewParams {
    uint32_t baseMipLevel;
    uint32_t mipLevels;
    uint32_t baseLayer;
    uint32_t layerCount;

    bool operator==(const ViewParams &other) const {
        return baseMipLevel == other.baseMipLevel
               && mipLevels == other.mipLevels
               && baseLayer == other.baseLayer
               && layerCount == other.layerCount;
    }
};

template<>
struct std::hash<ViewParams> {
    size_t operator()(ViewParams const &params) const noexcept {
        return (hash<uint32_t>()(params.mipLevels) >> 1) ^
               (hash<uint32_t>()(params.baseMipLevel) << 1) ^
               (hash<uint32_t>()(params.baseLayer) << 1) ^
               (hash<uint32_t>()(params.layerCount) << 1);
    }
};

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and are mostly suited for swap chain related logic.
 */
class Image {
protected:
    VmaAllocator allocator{};
    unique_ptr<VmaAllocation> allocation{};
    unique_ptr<vk::raii::Image> image;
    vk::Extent3D extent;
    vk::Format format{};
    uint32_t mipLevels;
    vk::ImageAspectFlags aspectMask;
    std::unordered_map<ViewParams, shared_ptr<vk::raii::ImageView> > cachedViews;

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
    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    getView(const RendererContext &ctx);

    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    getMipView(const RendererContext &ctx, uint32_t mipLevel);

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getLayerView(const RendererContext &ctx, uint32_t layer);

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getLayerMipView(const RendererContext &ctx, uint32_t layer, uint32_t mipLevel);

    [[nodiscard]] vk::Extent3D getExtent() const { return extent; }

    [[nodiscard]] vk::Extent2D getExtent2d() const { return {extent.width, extent.height}; }

    [[nodiscard]] vk::Format getFormat() const { return format; }

    [[nodiscard]] uint32_t getMipLevels() const { return mipLevels; }

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

    void saveToFile(const RendererContext &ctx, const std::filesystem::path &path) const;

protected:
    [[nodiscard]] shared_ptr<vk::raii::ImageView> getCachedView(const RendererContext &ctx, ViewParams params);
};

class CubeImage final : public Image {
    std::vector<unique_ptr<vk::raii::ImageView> > layerViews;
    std::vector<unique_ptr<vk::raii::ImageView> > attachmentLayerViews;
    std::vector<std::vector<unique_ptr<vk::raii::ImageView> > > layerMipViews; // layerMipViews[layer][mip]

public:
    explicit CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                       vk::MemoryPropertyFlags properties);

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getView(const RendererContext &ctx) override;

    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getMipView(const RendererContext &ctx, uint32_t mipLevel) override;

    void copyFromBuffer(vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer) override;

    void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                          const vk::raii::CommandBuffer &commandBuffer) const override;
};

class Texture {
    unique_ptr<Image> image;
    unique_ptr<vk::raii::Sampler> textureSampler;

    friend class TextureBuilder;

    Texture() = default;

public:
    [[nodiscard]] Image &getImage() const { return *image; }

    [[nodiscard]] const vk::raii::Sampler &getSampler() const { return *textureSampler; }

    [[nodiscard]] uint32_t getMipLevels() const { return image->getMipLevels(); }

    [[nodiscard]] vk::Format getFormat() const { return image->getFormat(); }

    void generateMipmaps(const RendererContext &ctx, vk::ImageLayout finalLayout) const;

private:
    void createSampler(const RendererContext &ctx, vk::SamplerAddressMode addressMode);
};

enum class SwizzleComponent {
    R,
    G,
    B,
    A,
    ZERO,
    ONE,
    MAX,
    HALF_MAX
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

    std::optional<std::array<SwizzleComponent, 4> > swizzle{
        {SwizzleComponent::R, SwizzleComponent::G, SwizzleComponent::B, SwizzleComponent::A}
    };

    vk::SamplerAddressMode addressMode = vk::SamplerAddressMode::eRepeat;

    std::optional<vk::Extent3D> desiredExtent;

    std::vector<std::filesystem::path> paths;
    void *memorySource = nullptr;
    bool isFromSwizzleFill = false;

    struct LoadedTextureData {
        std::vector<void *> sources;
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

    TextureBuilder &withSwizzle(std::array<SwizzleComponent, 4> sw);

    TextureBuilder &fromPaths(const std::vector<std::filesystem::path> &sources);

    TextureBuilder &fromMemory(void *ptr, vk::Extent3D extent);

    TextureBuilder &fromSwizzleFill(vk::Extent3D extent);

    [[nodiscard]] unique_ptr<Texture>
    create(const RendererContext &ctx) const;

private:
    void checkParams() const;

    [[nodiscard]] uint32_t getLayerCount() const;

    [[nodiscard]] LoadedTextureData loadFromPaths() const;

    [[nodiscard]] LoadedTextureData loadFromMemory() const;

    [[nodiscard]] LoadedTextureData loadFromSwizzleFill() const;

    [[nodiscard]] unique_ptr<Buffer> makeStagingBuffer(const RendererContext &ctx, const LoadedTextureData &data) const;

    static void *mergeChannels(const std::vector<void *> &channelsData, size_t textureSize, size_t componentCount);

    void performSwizzle(uint8_t *data, size_t size) const;
};

class RenderTarget {
    shared_ptr<vk::raii::ImageView> view;
    shared_ptr<vk::raii::ImageView> resolveView;
    vk::Format format{};

    vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear;
    vk::AttachmentStoreOp storeOp = vk::AttachmentStoreOp::eStore;

public:
    RenderTarget(shared_ptr<vk::raii::ImageView> view, vk::Format format);

    RenderTarget(shared_ptr<vk::raii::ImageView> view, shared_ptr<vk::raii::ImageView> resolveView, vk::Format format);

    RenderTarget(const RendererContext &ctx, const Texture &texture);

    [[nodiscard]] vk::Format getFormat() const { return format; }

    [[nodiscard]] vk::RenderingAttachmentInfo getAttachmentInfo() const;

    void overrideAttachmentConfig(vk::AttachmentLoadOp loadOp,
                                  vk::AttachmentStoreOp storeOp = vk::AttachmentStoreOp::eStore);
};

namespace vkutils::img {
    [[nodiscard]] vk::raii::ImageView
    createImageView(const RendererContext &ctx, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags,
                    uint32_t baseMipLevel = 0, uint32_t mipLevels = 1, uint32_t layer = 0);

    [[nodiscard]] vk::raii::ImageView
    createCubeImageView(const RendererContext &ctx, vk::Image image, vk::Format format,
                        vk::ImageAspectFlags aspectFlags, uint32_t baseMipLevel = 0, uint32_t mipLevels = 1);

    [[nodiscard]] bool isDepthFormat(vk::Format format);

    [[nodiscard]] size_t getFormatSizeInBytes(vk::Format format);
}

struct ImageBarrierInfo {
    vk::AccessFlagBits srcAccessMask;
    vk::AccessFlagBits dstAccessMask;
    vk::PipelineStageFlagBits srcStage;
    vk::PipelineStageFlagBits dstStage;
};

static const std::map<std::pair<vk::ImageLayout, vk::ImageLayout>, ImageBarrierInfo> transitionBarrierSchemes{
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
