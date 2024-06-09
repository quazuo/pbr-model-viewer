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

namespace utils::img {
    [[nodiscard]]
    std::unique_ptr<vk::raii::ImageView> createImageView(const RendererContext &ctx, vk::Image image,
                                                         vk::Format format, vk::ImageAspectFlags aspectFlags,
                                                         std::uint32_t mipLevels);

    void transitionImageLayout(const RendererContext &ctx, vk::Image image, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout, std::uint32_t mipLevels,
                               const vk::raii::CommandPool &cmdPool, const vk::raii::Queue &queue);
}
