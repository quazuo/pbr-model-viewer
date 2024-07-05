#include "swapchain.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "src/render/renderer.h"

SwapChainSupportDetails::SwapChainSupportDetails(const vk::raii::PhysicalDevice &physicalDevice,
                                                 const vk::raii::SurfaceKHR &surface)
    : capabilities(physicalDevice.getSurfaceCapabilitiesKHR(*surface)),
      formats(physicalDevice.getSurfaceFormatsKHR(*surface)),
      presentModes(physicalDevice.getSurfacePresentModesKHR(*surface)) {
}

SwapChain::SwapChain(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface,
                     const QueueFamilyIndices &queueFamilies, GLFWwindow *window,
                     vk::SampleCountFlagBits sampleCount) : msaaSampleCount(sampleCount) {
    const auto [capabilities, formats, presentModes] = SwapChainSupportDetails{*ctx.physicalDevice, surface};

    extent = chooseExtent(capabilities, window);

    const vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
    imageFormat = surfaceFormat.format;

    const vk::PresentModeKHR presentMode = choosePresentMode(presentModes);

    const auto &[graphicsComputeFamily, presentFamily] = queueFamilies;
    const uint32_t queueFamilyIndices[] = {graphicsComputeFamily.value(), presentFamily.value()};
    const bool isUniformFamily = graphicsComputeFamily == presentFamily;

    const vk::SwapchainCreateInfoKHR createInfo{
        .surface = *surface,
        .minImageCount = getImageCount(ctx, surface),
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = isUniformFamily ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
        .queueFamilyIndexCount = isUniformFamily ? 0u : 2u,
        .pQueueFamilyIndices = isUniformFamily ? nullptr : queueFamilyIndices,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = vk::True,
    };

    swapChain = make_unique<vk::raii::SwapchainKHR>(ctx.device->createSwapchainKHR(createInfo));
    images = swapChain->getImages();

    createImageViews(ctx);
    createColorResources(ctx);

    depthFormat = findDepthFormat(ctx);
    createDepthResources(ctx);
}

uint32_t SwapChain::getImageCount(const RendererContext &ctx, const vk::raii::SurfaceKHR &surface) {
    const auto [capabilities, formats, presentModes] = SwapChainSupportDetails{*ctx.physicalDevice, surface};

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    return imageCount;
}

std::pair<vk::Result, uint32_t> SwapChain::acquireNextImage(const vk::raii::Semaphore &semaphore) {
    try {
        const auto &[result, imageIndex] = swapChain->acquireNextImage(UINT64_MAX, *semaphore);
        currentImageIndex = imageIndex;
        return {result, imageIndex};

    } catch (...) {
        return {vk::Result::eErrorOutOfDateKHR, 0};
    }
}

vk::Extent2D SwapChain::chooseExtent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    const uint32_t actualExtentWidth = std::clamp(
        static_cast<uint32_t>(width),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    const uint32_t actualExtentHeight = std::clamp(
        static_cast<uint32_t>(height),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );

    return {
        actualExtentWidth,
        actualExtentHeight
    };
}

vk::SurfaceFormatKHR SwapChain::chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    if (availableFormats.empty()) {
        throw std::runtime_error("unexpected empty list of available formats");
    }

    for (const auto &availableFormat: availableFormats) {
        if (
            availableFormat.format == vk::Format::eB8G8R8A8Unorm
            && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
        ) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR SwapChain::choosePresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
    for (const auto &availablePresentMode: availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

void SwapChain::createImageViews(const RendererContext &ctx) {
    for (const auto &image: images) {
        auto imageView = utils::img::createImageView(
            ctx,
            image,
            imageFormat,
            vk::ImageAspectFlagBits::eColor,
            0,
            1,
            0
        );

        imageViews.push_back(std::move(imageView));
    }
}

void SwapChain::createFramebuffers(const RendererContext &ctx, const vk::raii::RenderPass &renderPass) {
    for (const auto &imageView: imageViews) {
        const std::array attachments = {*colorImage->getView(), *depthImage->getView(), **imageView};

        const vk::FramebufferCreateInfo createInfo{
            .renderPass = *renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        };

        auto framebuffer = make_unique<vk::raii::Framebuffer>(*ctx.device, createInfo);
        framebuffers.emplace_back(std::move(framebuffer));
    }
}

void SwapChain::createColorResources(const RendererContext &ctx) {
    const vk::Format colorFormat = imageFormat;

    const vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = colorFormat,
        .extent = {
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaaSampleCount,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    colorImage = make_unique<Image>(
        ctx,
        imageInfo,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageAspectFlagBits::eColor
    );

    colorImage->createViews(ctx);
}

void SwapChain::createDepthResources(const RendererContext &ctx) {
    const vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = depthFormat,
        .extent = {
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaaSampleCount,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    depthImage = make_unique<Image>(
        ctx,
        imageInfo,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ImageAspectFlagBits::eDepth
    );

    depthImage->createViews(ctx);
}

vk::Format SwapChain::findDepthFormat(const RendererContext &ctx) {
    return findSupportedFormat(
        ctx,
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

vk::Format SwapChain::findSupportedFormat(const RendererContext &ctx, const std::vector<vk::Format> &candidates,
                                          const vk::ImageTiling tiling, const vk::FormatFeatureFlags features) {
    for (const vk::Format format: candidates) {
        const vk::FormatProperties props = ctx.physicalDevice->getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}
