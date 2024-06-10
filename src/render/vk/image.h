#pragma once

#include <filesystem>
#include <variant>

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"

struct RendererContext;

/**
 * Abstraction over a Vulkan image, making it easier to manage by hiding all the Vulkan API calls.
 * These images are allocated using VMA and are mostly suited for swap chain related logic.
 */
class Image {
    VmaAllocator allocator{};
    std::unique_ptr<VmaAllocation> allocation{};
    std::unique_ptr<vk::raii::Image> image;
    std::unique_ptr<vk::raii::ImageView> view;
    vk::Extent3D extent;

public:
    explicit Image(const RendererContext &ctx, const vk::ImageCreateInfo &imageInfo,
                   vk::MemoryPropertyFlags properties);

    ~Image();

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

    void createView(const RendererContext &ctx, vk::Format format, vk::ImageAspectFlags aspectFlags,
                    std::uint32_t mipLevels);

    /**
     * Copies the contents of a given buffer to this image and waits until completion.
     *
     * @param ctx Renderer context.
     * @param buffer Buffer from which to copy.
     * @param cmdPool Command pool from which a single-time command buffer should be allocated.
     * @param queue Queue to which the commands should be submitted.
     */
    void copyFromBuffer(const RendererContext &ctx, vk::Buffer buffer, const vk::raii::CommandPool &cmdPool,
                        const vk::raii::Queue &queue);
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
    const vk::raii::Sampler &getSampler() const { return *textureSampler; }

    [[nodiscard]]
    const vk::raii::ImageView &getView() const { return image->getView(); }

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
    bool hasMipmaps = false;

    using ptr_source_t = std::pair<vk::Extent3D, void *>;
    std::variant<std::nullopt_t, std::filesystem::path, ptr_source_t> source = std::nullopt;

public:
    TextureBuilder &useFormat(const vk::Format f) {
        format = f;
        return *this;
    }

    TextureBuilder &useLayout(const vk::ImageLayout l) {
        layout = l;
        return *this;
    }

    TextureBuilder &useUsage(const vk::ImageUsageFlags u) {
        usage = u;
        return *this;
    }

    TextureBuilder &makeMipmaps() {
        hasMipmaps = true;
        return *this;
    }

    TextureBuilder &fromPath(const std::filesystem::path &path) {
        source = path;
        return *this;
    }

    TextureBuilder &fromDataPtr(vk::Extent3D extent, void *data) {
        source = std::make_pair(extent, data);
        return *this;
    }

    [[nodiscard]] Texture create(const RendererContext &ctx, const vk::raii::CommandPool &cmdPool,
                                 const vk::raii::Queue &queue) const;
};

namespace utils::img {
    [[nodiscard]]
    std::unique_ptr<vk::raii::ImageView> createImageView(const RendererContext &ctx, vk::Image image,
                                                         vk::Format format, vk::ImageAspectFlags aspectFlags,
                                                         std::uint32_t mipLevels);

    void transitionImageLayout(const RendererContext &ctx, vk::Image image, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout, std::uint32_t mipLevels,
                               const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue);

    [[nodiscard]]
    size_t getFormatSizeInBytes(vk::Format format);
}
