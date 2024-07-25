#pragma once

#include "image.h"
#include "src/render/libs.h"

/**
 * Helper structure holding details about supported features of the swap chain.
 */
struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;

    SwapChainSupportDetails(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::SurfaceKHR &surface);
};

struct RendererContext;
struct QueueFamilyIndices;
struct GLFWwindow;

struct SwapChainRenderTargets {
    RenderTarget colorTarget;
    RenderTarget depthTarget;
};

/**
* Abstraction over a Vulkan swap chain, making it easier to manage by hiding all the Vulkan API calls.
*/
class SwapChain {
    unique_ptr<vk::raii::SwapchainKHR> swapChain;
    std::vector<vk::Image> images;
    vk::Format imageFormat{};
    vk::Format depthFormat{};
    vk::Extent2D extent{};

    unique_ptr<Image> colorImage;
    unique_ptr<Image> depthImage;

    std::vector<shared_ptr<vk::raii::ImageView>> cachedViews;

    uint32_t currentImageIndex = 0;

    vk::SampleCountFlagBits msaaSampleCount;

public:
    explicit SwapChain(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface,
                       const QueueFamilyIndices &queueFamilies, GLFWwindow *window,
                       vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1);

    SwapChain(const SwapChain &other) = delete;

    SwapChain &operator=(const SwapChain &other) = delete;

    [[nodiscard]] const vk::raii::SwapchainKHR &operator*() const { return *swapChain; }

    [[nodiscard]] vk::Format getImageFormat() const { return imageFormat; }

    [[nodiscard]] vk::Format getDepthFormat() const { return depthFormat; }

    [[nodiscard]] vk::Extent2D getExtent() const { return extent; }

    /**
     * Returns the index of the image that was most recently acquired and will be presented next.
     * @return Index of the current image.
     */
    [[nodiscard]] uint32_t getCurrentImageIndex() const { return currentImageIndex; }

    /**
     * Wraps swapchain image views in `RenderTarget` objects and returns them.
     * When called the first time, these views are created and cached for later.
     */
    [[nodiscard]] std::vector<SwapChainRenderTargets> getRenderTargets(const RendererContext &ctx);

    /**
     * Requests a new image from the swap chain and signals a given semaphore when the image is available.
     * @param semaphore Semaphore which should be signalled after completion.
     * @return Result code and index of the new image.
     */
    [[nodiscard]] std::pair<vk::Result, uint32_t> acquireNextImage(const vk::raii::Semaphore &semaphore);

    [[nodiscard]] static uint32_t getImageCount(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface);

    /**
     * Records commands that transition the most newly acquired image to a layout
     * appropriate for having the image serve as a color attachment.
     */
    void transitionToAttachmentLayout(const vk::raii::CommandBuffer &commandBuffer) const;

    /**
     * Records commands that transition the most newly acquired image to a layout
     * appropriate for having the image be presented to the screen.
     */
    void transitionToPresentLayout(const vk::raii::CommandBuffer &commandBuffer) const;

private:
    void createColorResources(const RendererContext &ctx);

    void createDepthResources(const RendererContext &ctx);

    [[nodiscard]] static vk::Format findDepthFormat(const RendererContext &ctx);

    [[nodiscard]] static vk::Format
    findSupportedFormat(const RendererContext &ctx, const std::vector<vk::Format> &candidates,
                        vk::ImageTiling tiling, vk::FormatFeatureFlags features);

    [[nodiscard]] static vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window);

    static vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats);

    static vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes);
};
