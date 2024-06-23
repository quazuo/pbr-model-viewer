#pragma once

#include <filesystem>
#include <variant>

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"

class Buffer;
struct RendererContext;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and are mostly suited for swap chain related logic.
 */
class Image {
protected:
    VmaAllocator allocator{};
    std::unique_ptr<VmaAllocation> allocation{};
    std::unique_ptr<vk::raii::Image> image;
    std::unique_ptr<vk::raii::ImageView> view;
    vk::Extent3D extent;

public:
    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                   vk::MemoryPropertyFlags properties);

    virtual ~Image();

    Image(const Image &other) = delete;

    Image(Image &&other) = delete;

    Image &operator=(const Image &other) = delete;

    Image &operator=(Image &&other) = delete;

    /**
     * Returns a raw handle to the actual Vulkan image.
     * @return Handle to the image.
     */
    [[nodiscard]]
    const vk::raii::Image &get() const { return *image; }

    /**
     * Returns a raw handle to the actual Vulkan image view associated with this image.
     * @return Handle to the image view.
     */
    [[nodiscard]]
    const vk::raii::ImageView &getView() const;

    [[nodiscard]]
    vk::Extent3D getExtent() const { return extent; }

    virtual void createView(const RendererContext &ctx, vk::Format format, vk::ImageAspectFlags aspectFlags,
                            std::uint32_t mipLevels);

    /**
     * Copies the contents of a given buffer to this image and waits until completion.
     *
     * @param ctx Renderer context.
     * @param buffer Buffer from which to copy.
     * @param cmdPool Command pool from which a single-time command buffer should be allocated.
     * @param queue Queue to which the commands should be submitted.
     */
    virtual void copyFromBuffer(const RendererContext &ctx, vk::Buffer buffer, const vk::raii::CommandPool &cmdPool,
                                const vk::raii::Queue &queue);
};

class CubeImage final : public Image {
    std::vector<std::unique_ptr<vk::raii::ImageView> > layerViews;

public:
    explicit CubeImage(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                       vk::MemoryPropertyFlags properties);

    /**
     * Returns a raw handle to the actual Vulkan image view associated with a specific layer of this image.
     * @return Handle to the layer's image view.
     *
     * @param layerIndex Layer index to which the view should refer.
     */
    [[nodiscard]]
    const vk::raii::ImageView &getLayerView(const std::uint32_t layerIndex) const { return *layerViews[layerIndex]; }

    void createView(const RendererContext &ctx, vk::Format format, vk::ImageAspectFlags aspectFlags,
                    std::uint32_t mipLevels) override;

    void copyFromBuffer(const RendererContext &ctx, vk::Buffer buffer, const vk::raii::CommandPool &cmdPool,
                        const vk::raii::Queue &queue) override;
};

class Texture {
    std::unique_ptr<Image> image;
    std::unique_ptr<vk::raii::Sampler> textureSampler;
    uint32_t mipLevels{};
    vk::Format format{};

    friend class TextureBuilder;

    Texture() = default;

public:
    [[nodiscard]]
    const vk::raii::Image &getImage() const { return image->get(); }

    [[nodiscard]]
    const vk::raii::Sampler &getSampler() const { return *textureSampler; }

    [[nodiscard]]
    const vk::raii::ImageView &getView() const { return image->getView(); }

    /**
     * Returns a raw handle to the actual Vulkan image view associated with a specific layer of this image.
     * @return Handle to the layer's image view.
     *
     * @param layerIndex Layer index to which the view should refer.
     */
    [[nodiscard]]
    const vk::raii::ImageView &getLayerView(std::uint32_t layerIndex) const;

private:
    void generateMipmaps(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                         const vk::raii::Queue &queue, vk::ImageLayout finalLayout) const;

    void createSampler(const RendererContext &ctx);
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

    std::vector<std::filesystem::path> paths;

    struct LoadedTextureData {
        std::unique_ptr<Buffer> stagingBuffer;
        vk::Extent3D extent;
        std::uint32_t layerCount;
    };

public:
    TextureBuilder &useFormat(vk::Format f);

    TextureBuilder &useLayout(vk::ImageLayout l);

    TextureBuilder &useUsage(vk::ImageUsageFlags u);

    TextureBuilder &asCubemap();

    TextureBuilder &asSeparateChannels();

    TextureBuilder &asHdr();

    TextureBuilder &makeMipmaps();

    TextureBuilder &fromPaths(const std::vector<std::filesystem::path> &sources);

    [[nodiscard]]
    std::unique_ptr<Texture> create(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                                    const vk::raii::Queue &queue) const;

private:
    void checkParams() const;

    [[nodiscard]]
    LoadedTextureData loadFromPaths(const RendererContext &ctx) const;
};

namespace utils::img {
    [[nodiscard]]
    std::unique_ptr<vk::raii::ImageView> createImageView(const RendererContext &ctx, vk::Image image,
                                                         vk::Format format, vk::ImageAspectFlags aspectFlags,
                                                         std::uint32_t mipLevels, std::uint32_t layer);

    [[nodiscard]]
    std::unique_ptr<vk::raii::ImageView> createCubeImageView(const RendererContext &ctx, vk::Image image,
                                                             vk::Format format, vk::ImageAspectFlags aspectFlags,
                                                             std::uint32_t mipLevels);

    void transitionImageLayout(const RendererContext &ctx, vk::Image image, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout, std::uint32_t mipLevels, std::uint32_t layerCount,
                               const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue);

    [[nodiscard]]
    size_t getFormatSizeInBytes(vk::Format format);
}
