#pragma once

#include <filesystem>
#include <map>

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"
#include "src/render/globals.h"

class Buffer;

struct RendererContext;

/**
 * Parameters defining which mip levels and layers of a given image are available for a given view.
 * This struct is used mainly for caching views to eliminate creating multiple identical views.
 */
struct ViewParams {
    uint32_t baseMipLevel;
    uint32_t mipLevels;
    uint32_t baseLayer;
    uint32_t layerCount;

    // `unordered_map` requirement
    bool operator==(const ViewParams &other) const {
        return baseMipLevel == other.baseMipLevel
               && mipLevels == other.mipLevels
               && baseLayer == other.baseLayer
               && layerCount == other.layerCount;
    }
};

// `unordered_map` requirement
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
 * These images are allocated using VMA and as such are not suited for swap chain images.
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
     * Returns an image view containing all mip levels and all layers of this image.
     */
    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    getView(const RendererContext &ctx);

    /**
     * Returns an image view containing a single mip level and all layers of this image.
     */
    [[nodiscard]] virtual shared_ptr<vk::raii::ImageView>
    getMipView(const RendererContext &ctx, uint32_t mipLevel);

    /**
     * Returns an image view containing all mip levels and a single specified layer of this image.
     */
    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getLayerView(const RendererContext &ctx, uint32_t layer);

    /**
     * Returns an image view containing a single mip level and a single specified layer of this image.
     */
    [[nodiscard]] shared_ptr<vk::raii::ImageView>
    getLayerMipView(const RendererContext &ctx, uint32_t layer, uint32_t mipLevel);

    [[nodiscard]] vk::Extent3D getExtent() const { return extent; }

    [[nodiscard]] vk::Extent2D getExtent2d() const { return {extent.width, extent.height}; }

    [[nodiscard]] vk::Format getFormat() const { return format; }

    [[nodiscard]] uint32_t getMipLevels() const { return mipLevels; }

    /**
     * Records commands that copy the contents of a given buffer to this image.
     */
    virtual void copyFromBuffer(vk::Buffer buffer, const vk::raii::CommandBuffer &commandBuffer);

    /**
     * Records commands that transition this image's layout.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    virtual void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                                  const vk::raii::CommandBuffer &commandBuffer) const;

    /**
     * Records commands that transition this image's layout, also specifying a specific subresource range
     * on which the transition should occur.
     * A valid old layout must be provided, as the image's current layout is not being tracked.
     */
    void transitionLayout(vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                          vk::ImageSubresourceRange range, const vk::raii::CommandBuffer &commandBuffer) const;

    /**
     * Writes the contents of this image to a file on a given path.
     *
     * Disclaimer: this might not work very well as it wasn't tested very well
     * (nor do I care about it working perfectly) and was created purely to debug a single thing in the past.
     * However, I'm not removing this as I might use it (and make it work better) again in the future.
     */
    void saveToFile(const RendererContext &ctx, const std::filesystem::path &path) const;

protected:
    /**
     * Checks if a given view is cached already and if so, returns it without creating a new one.
     * Otherwise, creates the view and caches it for later.
     */
    [[nodiscard]] shared_ptr<vk::raii::ImageView> getCachedView(const RendererContext &ctx, ViewParams params);
};

class CubeImage final : public Image {
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
    unique_ptr<vk::raii::Sampler> sampler;

    friend class TextureBuilder;

    Texture() = default;

public:
    [[nodiscard]] Image &getImage() const { return *image; }

    [[nodiscard]] const vk::raii::Sampler &getSampler() const { return *sampler; }

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

/**
 * Builder used to streamline texture creation due to a huge amount of different parameters.
 * Currently only some specific scenarios are supported and some parameter combinations
 * might not be implemented, due to them not being needed at the moment.
 */
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

    /**
     * Designates the texture's contents to be initialized with data stored in a given file.
     * This requires 6 different paths for cubemap textures.
     */
    TextureBuilder &fromPaths(const std::vector<std::filesystem::path> &sources);

    /**
     * Designates the texture's contents to be initialized with data stored in memory.
     */
    TextureBuilder &fromMemory(void *ptr, vk::Extent3D extent);

    /**
     * Designates the texture's contents to be initialized with static data defined using `withSwizzle`.
     */
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

/**
 * Convenience wrapper around image views which are used as render targets.
 * This is primarily an abstraction to unify textures and swapchain images, so that they're used
 * in an uniform way.
 */
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

/**
 * List of stages and access masks for image layout transitions.
 * Currently there's no need for more fine-grained customization of these parameters during transitions,
 * so they're defined statically and used depeneding on the transition's start and end layouts.
 */
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
