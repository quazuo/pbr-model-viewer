#include "renderer.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <set>
#include <vector>
#include <fstream>
#include <filesystem>
#include <array>
#include <random>

#include "deps/tinyobjloader/tiny_obj_loader.h"
#include "src/utils/octree-gen.h"

#include "vertex.h"
#include "vk/cmd.h"

VmaAllocatorWrapper::VmaAllocatorWrapper(const vk::PhysicalDevice physicalDevice, const vk::Device device,
                                         const vk::Instance instance) {
    static constexpr VmaVulkanFunctions funcs{
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr
    };

    const VmaAllocatorCreateInfo allocatorCreateInfo{
        .physicalDevice = physicalDevice,
        .device = device,
        .pVulkanFunctions = &funcs,
        .instance = instance,
    };

    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}

VmaAllocatorWrapper::~VmaAllocatorWrapper() {
    vmaDestroyAllocator(allocator);
}

AutomatonRenderer::AutomatonRenderer(const AutomatonConfig &config) : automatonConfig(config) {
    constexpr int INIT_WINDOW_WIDTH = 1200;
    constexpr int INIT_WINDOW_HEIGHT = 800;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "3D Cellular Automata Viewer", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    camera = make_unique<Camera>(window);

    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();

    ctx.allocator = make_unique<VmaAllocatorWrapper>(**ctx.physicalDevice, **ctx.device, **instance);

    swapChain = make_unique<SwapChain>(ctx, *surface, findQueueFamilies(*ctx.physicalDevice), window, msaaSampleCount);

    createRenderPass();

    createDescriptorSetLayouts();

    createGraphicsPipeline();
    createComputePipeline();

    createCommandPool();

    swapChain->createFramebuffers(ctx, *renderPass);

    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createShaderStorageBuffers();

    createDescriptorPool();
    createDescriptorSets();

    createCommandBuffers();

    createSyncObjects();

    initImgui();
}

AutomatonRenderer::~AutomatonRenderer() {
    glfwDestroyWindow(window);
}

void AutomatonRenderer::updateAutomatonConfig(const AutomatonConfig &config) {
    automatonConfig = config;

    updateComputeUniformBuffers();
}

void AutomatonRenderer::setIsCursorLocked(const bool b) const {
    camera->setIsCursorLocked(b);
    glfwSetInputMode(window, GLFW_CURSOR, b ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void AutomatonRenderer::framebufferResizeCallback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto app = static_cast<AutomatonRenderer *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

// ==================== instance creation ====================

void AutomatonRenderer::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "3D Cellular Automata Viewer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3
    };

    const auto extensions = getRequiredExtensions();
    const vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo = makeDebugMessengerCreateInfo();

    const vk::InstanceCreateInfo createInfo{
        .pNext = enableValidationLayers ? &debugCreateInfo : nullptr,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<std::uint32_t>(enableValidationLayers ? validationLayers.size() : 0),
        .ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance = make_unique<vk::raii::Instance>(vkCtx, createInfo);
}

std::vector<const char *> AutomatonRenderer::getRequiredExtensions() {
    std::uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// ==================== validation layers ====================

bool AutomatonRenderer::checkValidationLayerSupport() {
    std::uint32_t layerCount;
    if (vk::enumerateInstanceLayerProperties(&layerCount, nullptr) != vk::Result::eSuccess) {
        throw std::runtime_error("couldn't fetch the number of instance layers!");
    }

    std::vector<vk::LayerProperties> availableLayers(layerCount);
    if (vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data()) != vk::Result::eSuccess) {
        throw std::runtime_error("couldn't fetch the instance layer properties!");
    }

    for (const char *layerName: validationLayers) {
        bool layerFound = false;

        for (const auto &layerProperties: availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

vk::DebugUtilsMessengerCreateInfoEXT AutomatonRenderer::makeDebugMessengerCreateInfo() {
    return {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                           | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                           | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                       | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                       | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback,
    };
}

void AutomatonRenderer::setupDebugMessenger() {
    if constexpr (!enableValidationLayers) return;

    const vk::DebugUtilsMessengerCreateInfoEXT createInfo = makeDebugMessengerCreateInfo();
    debugMessenger = make_unique<vk::raii::DebugUtilsMessengerEXT>(*instance, createInfo);
}

VKAPI_ATTR VkBool32 VKAPI_CALL AutomatonRenderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData
) {
    auto &stream = messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                       ? std::cout
                       : std::cerr;

    stream << "Validation layer:";
    stream << "\n\tSeverity: ";

    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            stream << "Verbose";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            stream << "Info";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            stream << "Warning";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            stream << "Error";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
            break;
    }

    stream << "\n\tMessage:" << pCallbackData->pMessage << std::endl;
    return vk::False;
}

// ==================== window surface ====================

void AutomatonRenderer::createSurface() {
    VkSurfaceKHR _surface;

    if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    surface = make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
}

// ==================== physical device ====================

void AutomatonRenderer::pickPhysicalDevice() {
    const std::vector<vk::raii::PhysicalDevice> devices = instance->enumeratePhysicalDevices();

    for (const auto &dev: devices) {
        if (isDeviceSuitable(dev)) {
            ctx.physicalDevice = make_unique<vk::raii::PhysicalDevice>(dev);
            msaaSampleCount = getMaxUsableSampleCount();
            usePyramidAcceleration = checkDeviceSubgroupSupport(*ctx.physicalDevice);
            return;
        }
    }

    throw std::runtime_error("failed to find a suitable GPU!");
}

[[nodiscard]]
bool AutomatonRenderer::isDeviceSuitable(const vk::raii::PhysicalDevice &physicalDevice) const {
    if (!findQueueFamilies(physicalDevice).isComplete()) {
        return false;
    }

    if (!checkDeviceExtensionSupport(physicalDevice)) {
        return false;
    }

    const SwapChainSupportDetails swapChainSupport = SwapChainSupportDetails{physicalDevice, *surface};
    if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty()) {
        return false;
    }

    const vk::PhysicalDeviceFeatures supportedFeatures = physicalDevice.getFeatures();
    if (!supportedFeatures.samplerAnisotropy) {
        return false;
    }

    const vk::PhysicalDeviceVulkan12Features features12 = physicalDevice
            .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features>()
            .get<vk::PhysicalDeviceVulkan12Features>();

    if (!features12.shaderInt8 || !features12.uniformAndStorageBuffer8BitAccess || !features12.
        storageBuffer8BitAccess) {
        return false;
    }

    return true;
}

[[nodiscard]]
QueueFamilyIndices AutomatonRenderer::findQueueFamilies(const vk::raii::PhysicalDevice &physicalDevice) const {
    const std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

    std::optional<std::uint32_t> graphicsComputeFamily;
    std::optional<std::uint32_t> presentFamily;

    std::uint32_t i = 0;
    for (const auto &queueFamily: queueFamilies) {
        const bool hasGraphicsSupport = static_cast<bool>(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics);
        const bool hasComputeSupport = static_cast<bool>(queueFamily.queueFlags & vk::QueueFlagBits::eCompute);
        if (hasGraphicsSupport && hasComputeSupport) {
            graphicsComputeFamily = i;
        }

        if (physicalDevice.getSurfaceSupportKHR(i, **surface)) {
            presentFamily = i;
        }

        if (graphicsComputeFamily.has_value() && presentFamily.has_value()) {
            break;
        }

        i++;
    }

    return {
        .graphicsComputeFamily = graphicsComputeFamily,
        .presentFamily = presentFamily
    };
}

bool AutomatonRenderer::checkDeviceExtensionSupport(const vk::raii::PhysicalDevice &physicalDevice) {
    const std::vector<vk::ExtensionProperties> availableExtensions =
            physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto &extension: availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

bool AutomatonRenderer::checkDeviceSubgroupSupport(const vk::raii::PhysicalDevice &physicalDevice) {
    const vk::PhysicalDeviceSubgroupProperties subgroupProperties = physicalDevice
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>()
            .get<vk::PhysicalDeviceSubgroupProperties>();

    if (!(subgroupProperties.supportedStages & vk::ShaderStageFlagBits::eCompute)) {
        return false;
    }

    if (!(subgroupProperties.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle)) {
        return false;
    }

    if (subgroupProperties.subgroupSize < 8) {
        return false;
    }

    return true;
}

// ==================== logical device ====================

void AutomatonRenderer::createLogicalDevice() {
    const auto [graphicsComputeFamily, presentFamily] = findQueueFamilies(*ctx.physicalDevice);
    const std::set uniqueQueueFamilies = {graphicsComputeFamily.value(), presentFamily.value()};

    constexpr float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    for (std::uint32_t queueFamily: uniqueQueueFamilies) {
        const vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = queueFamily,
            .queueCount = 1U,
            .pQueuePriorities = &queuePriority
        };
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{
        .samplerAnisotropy = vk::True,
    };

    vk::PhysicalDeviceSynchronization2FeaturesKHR sync2Features{
        .synchronization2 = vk::True,
    };

    vk::PhysicalDeviceVulkan12Features vulkan12Features{
        .pNext = &sync2Features,
        .storageBuffer8BitAccess = vk::True,
        .uniformAndStorageBuffer8BitAccess = vk::True,
        .shaderInt8 = vk::True,
        .timelineSemaphore = vk::True,
    };

    const vk::DeviceCreateInfo createInfo{
        .pNext = &vulkan12Features,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<std::uint32_t>(enableValidationLayers ? validationLayers.size() : 0),
        .ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures,
    };

    ctx.device = make_unique<vk::raii::Device>(*ctx.physicalDevice, createInfo);

    graphicsQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(graphicsComputeFamily.value(), 0));
    computeQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(graphicsComputeFamily.value(), 0));
    presentQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(presentFamily.value(), 0));
}

// ==================== swapchain ====================

void AutomatonRenderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    ctx.device->waitIdle();

    swapChain = {};
    swapChain =
            std::make_unique<SwapChain>(ctx, *surface, findQueueFamilies(*ctx.physicalDevice), window, msaaSampleCount);
    swapChain->createFramebuffers(ctx, *renderPass);
}

// ==================== descriptors ====================

void AutomatonRenderer::createDescriptorSetLayouts() {
    createGraphicsDescriptorSetLayouts();
    createComputeDescriptorSetLayout();
}

void AutomatonRenderer::createGraphicsDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding ssboLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array frameSetBindings{
        uboLayoutBinding,
    };

    static constexpr std::array ssboSetBindings{
        ssboLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo frameSetLayoutInfo{
        .bindingCount = static_cast<std::uint32_t>(frameSetBindings.size()),
        .pBindings = frameSetBindings.data(),
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo ssboSetLayoutInfo{
        .bindingCount = static_cast<std::uint32_t>(ssboSetBindings.size()),
        .pBindings = ssboSetBindings.data(),
    };

    graphicsDescriptorSetLayouts = {
        .frameSetLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, frameSetLayoutInfo),
        .ssboSetLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, ssboSetLayoutInfo),
    };
}

void AutomatonRenderer::createComputeDescriptorSetLayout() {
    static constexpr vk::DescriptorSetLayoutBinding uboBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    };

    static constexpr vk::DescriptorSetLayoutBinding prevFrameSsboBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    };

    static constexpr vk::DescriptorSetLayoutBinding currFrameSsboBinding{
        .binding = 2U,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    };

    static constexpr std::array bindings{
        uboBinding,
        prevFrameSsboBinding,
        currFrameSsboBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    computeDescriptorSetLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, layoutInfo);
}

void AutomatonRenderer::createDescriptorPool() {
    static constexpr vk::DescriptorPoolSize uboPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT + AUTOMATON_RESOURCE_COUNT),
    };

    static constexpr vk::DescriptorPoolSize ssboPoolSize{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = static_cast<std::uint32_t>(AUTOMATON_RESOURCE_COUNT) * 3,
    };

    static constexpr std::array poolSizes = {uboPoolSize, ssboPoolSize};

    static constexpr vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT + AUTOMATON_RESOURCE_COUNT * 2),
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);
}

void AutomatonRenderer::createDescriptorSets() {
    createGraphicsDescriptorSets();
    createComputeDescriptorSets();
}

void AutomatonRenderer::createGraphicsDescriptorSets() {
    constexpr std::uint32_t frameSetsCount = MAX_FRAMES_IN_FLIGHT;
    constexpr std::uint32_t ssboSetsCount = AUTOMATON_RESOURCE_COUNT;

    const std::vector frameSetLayouts(frameSetsCount, **graphicsDescriptorSetLayouts.frameSetLayout);

    const vk::DescriptorSetAllocateInfo frameSetAllocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = frameSetsCount,
        .pSetLayouts = frameSetLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> frameDescriptorSets = ctx.device->allocateDescriptorSets(frameSetAllocInfo);

    for (size_t i = 0; i < frameSetsCount; i++) {
        const vk::DescriptorBufferInfo uboBufferInfo{
            .buffer = frameResources[i].graphicsUniformBuffer->get(),
            .offset = 0U,
            .range = sizeof(GraphicsUBO),
        };

        const vk::WriteDescriptorSet uboDescriptorWrite{
            .dstSet = *frameDescriptorSets[i],
            .dstBinding = 0U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uboBufferInfo
        };

        const std::array descriptorWrites = {uboDescriptorWrite};
        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        frameResources[i].graphicsDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(frameDescriptorSets[i]));
    }

    const std::vector ssboSetLayouts(ssboSetsCount, **graphicsDescriptorSetLayouts.ssboSetLayout);

    const vk::DescriptorSetAllocateInfo ssboSetAllocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = ssboSetsCount,
        .pSetLayouts = ssboSetLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> ssboDescriptorSets = ctx.device->allocateDescriptorSets(ssboSetAllocInfo);

    for (size_t i = 0; i < ssboSetsCount; i++) {
        const vk::DeviceSize bufferSize = OctreeGen::getOctreeBufferSize(automatonConfig.gridDepth);

        const vk::DescriptorBufferInfo ssboBufferInfo{
            .buffer = automatonResources[i].shaderStorageBuffer->get(),
            .offset = 0U,
            .range = bufferSize,
        };

        const vk::WriteDescriptorSet ssboDescriptorWrite{
            .dstSet = *ssboDescriptorSets[i],
            .dstBinding = 1U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &ssboBufferInfo
        };

        const std::array descriptorWrites = {ssboDescriptorWrite};
        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        automatonResources[i].graphicsDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(ssboDescriptorSets[i]));
    }
}

void AutomatonRenderer::createComputeDescriptorSets() {
    constexpr std::uint32_t setsCount = AUTOMATON_RESOURCE_COUNT;

    const std::vector layouts(setsCount, **computeDescriptorSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = layouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        const vk::DescriptorBufferInfo uboInfo{
            .buffer = automatonResources[i].computeUniformBuffer->get(),
            .offset = 0U,
            .range = sizeof(ComputeUBO),
        };

        const vk::WriteDescriptorSet uboDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 0U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uboInfo
        };

        const vk::DeviceSize bufferSize = OctreeGen::getOctreeBufferSize(automatonConfig.gridDepth);

        const size_t prevSsboIndex = i == 0 ? AUTOMATON_RESOURCE_COUNT - 1 : i - 1;

        const vk::DescriptorBufferInfo prevFrameSsboInfo{
            .buffer = automatonResources[prevSsboIndex].shaderStorageBuffer->get(),
            .offset = 0U,
            .range = bufferSize,
        };

        const vk::WriteDescriptorSet prevFrameSsboDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 1U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &prevFrameSsboInfo
        };

        const vk::DescriptorBufferInfo currFrameSsboInfo{
            .buffer = automatonResources[i].shaderStorageBuffer->get(),
            .offset = 0U,
            .range = bufferSize,
        };

        const vk::WriteDescriptorSet currFrameSsboDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 2U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &currFrameSsboInfo
        };

        const std::array descriptorWrites = {
            uboDescriptorWrite,
            prevFrameSsboDescriptorWrite,
            currFrameSsboDescriptorWrite
        };

        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        automatonResources[i].computeDescriptorSet = make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[i]));
    }
}

// ==================== graphics pipeline ====================

void AutomatonRenderer::createRenderPass() {
    const vk::AttachmentDescription colorAttachment{
        .format = swapChain->getImageFormat(),
        .samples = msaaSampleCount,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    const vk::AttachmentDescription depthAttachment{
        .format = swapChain->getDepthFormat(),
        .samples = msaaSampleCount,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    const vk::AttachmentDescription colorAttachmentResolve{
        .format = swapChain->getImageFormat(),
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR,
    };

    static constexpr vk::AttachmentReference colorAttachmentRef{
        .attachment = 0U,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    static constexpr vk::AttachmentReference depthAttachmentRef{
        .attachment = 1U,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    };

    static constexpr vk::AttachmentReference colorAttachmentResolveRef{
        .attachment = 2U,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    static constexpr vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = &colorAttachmentResolveRef,
        .pDepthStencilAttachment = &depthAttachmentRef,
    };

    const std::array attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};

    static constexpr vk::SubpassDependency dependency{
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0U,
        .srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                        | vk::PipelineStageFlagBits::eLateFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                        | vk::PipelineStageFlagBits::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                         | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    };

    const vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };

    renderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
}

void AutomatonRenderer::createGraphicsPipeline() {
    const vk::raii::ShaderModule vertShaderModule = createShaderModule("../shaders/vert.spv");
    const vk::raii::ShaderModule fragShaderModule = createShaderModule("../shaders/frag.spv");

    const vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = *vertShaderModule,
        .pName = "main",
    };

    const vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = *fragShaderModule,
        .pName = "main",
    };

    const vk::PipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageInfo,
        fragShaderStageInfo,
    };

    const auto bindingDescription = Vertex::getBindingDescription();
    const auto attributeDescriptions = Vertex::getAttributeDescriptions();

    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    static constexpr std::array dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    static constexpr vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    constexpr vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1U,
        .scissorCount = 1U,
    };

    constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSampleCount,
        .sampleShadingEnable = vk::False,
        .minSampleShading = 1.0f,
    };

    static constexpr vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA,
    };

    static constexpr vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .attachmentCount = 1u,
        .pAttachments = &colorBlendAttachment,
    };

    static constexpr vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
    };

    const std::array descriptorSetLayouts = {
        **graphicsDescriptorSetLayouts.frameSetLayout,
        **graphicsDescriptorSetLayouts.ssboSetLayout,
    };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = descriptorSetLayouts.size(),
        .pSetLayouts = descriptorSetLayouts.data(),
    };

    graphicsPipelineLayout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = **graphicsPipelineLayout,
        .renderPass = **renderPass,
        .subpass = 0,
    };

    graphicsPipeline = make_unique<vk::raii::Pipeline>(*ctx.device, nullptr, pipelineInfo);
}

void AutomatonRenderer::createComputePipeline() {
    const vk::raii::ShaderModule compShaderModule = createShaderModule("../shaders/comp.spv");

    const vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *compShaderModule,
        .pName = "main",
    };

    static constexpr vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(ComputePushConstants),
    };

    const vk::PipelineLayoutCreateInfo layoutInfo{
        .setLayoutCount = 1U,
        .pSetLayouts = &**computeDescriptorSetLayout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &pushConstantRange,
    };

    computePipelineLayout = make_unique<vk::raii::PipelineLayout>(*ctx.device, layoutInfo);

    const vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = computeShaderStageInfo,
        .layout = **computePipelineLayout,
    };

    computePipeline = make_unique<vk::raii::Pipeline>(*ctx.device, nullptr, pipelineInfo);
}

[[nodiscard]]
vk::raii::ShaderModule AutomatonRenderer::createShaderModule(const std::filesystem::path &path) const {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    const size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    const vk::ShaderModuleCreateInfo createInfo{
        .codeSize = buffer.size(),
        .pCode = reinterpret_cast<const std::uint32_t *>(buffer.data()),
    };

    return {*ctx.device, createInfo};
}

// ==================== multisampling ====================

[[nodiscard]]
vk::SampleCountFlagBits AutomatonRenderer::getMaxUsableSampleCount() const {
    const vk::PhysicalDeviceProperties physicalDeviceProperties = ctx.physicalDevice->getProperties();

    const vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
                                        & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
}

// ==================== buffers ====================

/**
 * The only thing we conventionally render is the screen-filling quad on which
 * we will render the ray-marched graphics.
 */
static constexpr std::array<Vertex, 4> quadVertices = {
    {
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        {{1.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
        {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f}}
    }
};

static constexpr std::array<std::uint32_t, 6> quadIndices = {
    0, 1, 2, 2, 3, 0
};

void AutomatonRenderer::createVertexBuffer() {
    constexpr vk::DeviceSize bufferSize = sizeof(quadVertices[0]) * quadVertices.size();

    Buffer stagingBuffer{
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();
    memcpy(data, quadVertices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer.unmap();

    vertexBuffer = std::make_unique<Buffer>(
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    copyBuffer(stagingBuffer.get(), vertexBuffer->get(), bufferSize);
}

void AutomatonRenderer::createIndexBuffer() {
    constexpr vk::DeviceSize bufferSize = sizeof(quadIndices[0]) * quadIndices.size();

    Buffer stagingBuffer{
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();
    memcpy(data, quadIndices.data(), static_cast<size_t>(bufferSize));
    stagingBuffer.unmap();

    indexBuffer = std::make_unique<Buffer>(
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    copyBuffer(stagingBuffer.get(), indexBuffer->get(), bufferSize);
}

void AutomatonRenderer::createUniformBuffers() {
    for (auto &res: frameResources) {
        res.graphicsUniformBuffer = std::make_unique<Buffer>(
            ctx.allocator->get(),
            sizeof(GraphicsUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        res.graphicsUboMapped = res.graphicsUniformBuffer->map();
    }

    for (auto &res: automatonResources) {
        res.computeUniformBuffer = std::make_unique<Buffer>(
            ctx.allocator->get(),
            sizeof(ComputeUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        res.computeUboMapped = res.computeUniformBuffer->map();
    }
}

void AutomatonRenderer::createShaderStorageBuffers() {
    const vk::DeviceSize bufferSize = OctreeGen::getOctreeBufferSize(automatonConfig.gridDepth);

    for (auto &res: automatonResources) {
        if (res.shaderStorageBuffer) {
            res.shaderStorageBuffer = {};
        }

        res.shaderStorageBuffer = make_unique<Buffer>(
            ctx.allocator->get(),
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst
            | vk::BufferUsageFlagBits::eVertexBuffer
            | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    }
}

void AutomatonRenderer::copyBuffer(const vk::Buffer srcBuffer, const vk::Buffer dstBuffer,
                                   const vk::DeviceSize size) const {
    const vk::raii::CommandBuffer commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    const vk::BufferCopy copyRegion{
        .srcOffset = 0U,
        .dstOffset = 0U,
        .size = size,
    };

    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);
}

void AutomatonRenderer::fillSsbos(const OctreeGen::OctreeBuf &initValues) const {
    waitIdle();

    const vk::DeviceSize bufferSize = initValues.size();

    Buffer stagingBuffer{
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();
    memcpy(data, initValues.data(), static_cast<size_t>(bufferSize));
    stagingBuffer.unmap();

    for (auto &res: automatonResources) {
        copyBuffer(stagingBuffer.get(), res.shaderStorageBuffer->get(), bufferSize);
    }
}

void AutomatonRenderer::rebuildSsbos() {
    waitIdle();

    createShaderStorageBuffers();

    // move old descriptor pool here before creating a new one so that it doesn't destruct.
    // it cannot destruct yet because `createDescriptorSets()` frees old sets and freeing them needs the pool.
    const auto oldPool = std::move(*descriptorPool);
    createDescriptorPool();
    createDescriptorSets();
}

// ==================== commands ====================

void AutomatonRenderer::createCommandPool() {
    const QueueFamilyIndices queueFamilyIndices = findQueueFamilies(*ctx.physicalDevice);

    const vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.graphicsComputeFamily.value()
    };

    commandPool = make_unique<vk::raii::CommandPool>(*ctx.device, poolInfo);
}

void AutomatonRenderer::createCommandBuffers() {
    const vk::CommandBufferAllocateInfo primaryAllocInfo{
        .commandPool = **commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<std::uint32_t>(frameResources.size()),
    };

    const vk::CommandBufferAllocateInfo secondaryAllocInfo{
        .commandPool = **commandPool,
        .level = vk::CommandBufferLevel::eSecondary,
        .commandBufferCount = static_cast<std::uint32_t>(frameResources.size()),
    };

    vk::raii::CommandBuffers graphicsCommandBuffers{*ctx.device, primaryAllocInfo};
    vk::raii::CommandBuffers guiCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers computeCommandBuffers{*ctx.device, primaryAllocInfo};

    for (size_t i = 0; i < graphicsCommandBuffers.size(); i++) {
        frameResources[i].graphicsCmdBuf =
                make_unique<vk::raii::CommandBuffer>(std::move(graphicsCommandBuffers[i]));
        frameResources[i].guiCmdBuf =
                make_unique<vk::raii::CommandBuffer>(std::move(guiCommandBuffers[i]));
        frameResources[i].computeCmdBuf =
                make_unique<vk::raii::CommandBuffer>(std::move(computeCommandBuffers[i]));
    }
}

void AutomatonRenderer::recordGraphicsCommandBuffer() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].graphicsCmdBuf;

    constexpr vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);

    constexpr vk::ClearColorValue clearColor{0.0f, 0.0f, 0.0f, 1.0f};

    constexpr std::array<vk::ClearValue, 2> clearValues{
        {
            clearColor,
            vk::ClearDepthStencilValue{
                .depth = 1.0f,
                .stencil = 0,
            }
        }
    };

    const vk::Extent2D swapChainExtent = swapChain->getExtent();

    const vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = **renderPass,
        .framebuffer = *swapChain->getCurrentFramebuffer(),
        .renderArea = {
            .offset = {0, 0},
            .extent = swapChainExtent
        },
        .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
    };

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInlineAndSecondaryCommandBuffersEXT);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **graphicsPipeline);

    static constexpr std::array<vk::DeviceSize, 1> offsets{};
    commandBuffer.bindVertexBuffers(0, vertexBuffer->get(), offsets);
    commandBuffer.bindIndexBuffer(indexBuffer->get(), 0, vk::IndexType::eUint32);

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = static_cast<float>(swapChainExtent.height), // flip the y-axis
        .width = static_cast<float>(swapChainExtent.width),
        .height = -1 * static_cast<float>(swapChainExtent.height), // flip the y-axis
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    commandBuffer.setViewport(0, viewport);

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainExtent
    };

    commandBuffer.setScissor(0, scissor);

    const std::array descriptorSets = {
        **frameResources[currentFrameIdx].graphicsDescriptorSet,
        **automatonResources[mostRecentSsboIdx].graphicsDescriptorSet
    };

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        **graphicsPipelineLayout,
        0,
        descriptorSets,
        nullptr
    );

    commandBuffer.drawIndexed(static_cast<std::uint32_t>(quadIndices.size()), 1, 0, 0, 0);

    commandBuffer.executeCommands(**frameResources[currentFrameIdx].guiCmdBuf);

    commandBuffer.endRenderPass();

    commandBuffer.end();
}

void AutomatonRenderer::recordComputeCommandBuffer() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].computeCmdBuf;

    constexpr vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **computePipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        **computePipelineLayout,
        0,
        **automatonResources[mostRecentSsboIdx].computeDescriptorSet,
        nullptr
    );

    static constexpr vk::MemoryBarrier2 memoryBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead
    };

    static constexpr vk::DependencyInfo dependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memoryBarrier
    };

    for (int level = automatonConfig.gridDepth - 1; level >= 0; level--) {
        const std::uint32_t pyramidHeight =
                usePyramidAcceleration && level != automatonConfig.gridDepth - 1 && level > 0
                    ? 2
                    : 1;

        const ComputePushConstants consts{
            .level = static_cast<std::uint32_t>(level),
            .pyramidHeight = pyramidHeight,
        };

        commandBuffer.pushConstants<ComputePushConstants>(
            **computePipelineLayout,
            vk::ShaderStageFlagBits::eCompute,
            0,
            consts
        );

        const std::uint32_t levelWidth = OctreeGen::getOctreeLevelWidth(level);

        static_assert(WORK_GROUP_SIZE.x == WORK_GROUP_SIZE.y);
        static_assert(WORK_GROUP_SIZE.x == WORK_GROUP_SIZE.z);

        commandBuffer.dispatch(
            std::max(levelWidth / WORK_GROUP_SIZE.x, 1u),
            std::max(levelWidth / WORK_GROUP_SIZE.y, 1u),
            std::max(levelWidth / WORK_GROUP_SIZE.z, 1u)
        );

        commandBuffer.pipelineBarrier2(dependencyInfo);

        level -= consts.pyramidHeight - 1;
    }

    commandBuffer.pipelineBarrier2(dependencyInfo);

    commandBuffer.end();
}

// ==================== sync ====================

void AutomatonRenderer::createSyncObjects() {
    static constexpr vk::SemaphoreTypeCreateInfo typeCreateInfo{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };

    constexpr vk::SemaphoreCreateInfo timelineSemaphoreInfo{
        .pNext = &typeCreateInfo
    };

    constexpr vk::SemaphoreCreateInfo binarySemaphoreInfo;

    constexpr vk::FenceCreateInfo fenceInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled
    };

    for (auto &res: frameResources) {
        res.sync = {
            .imageAvailableSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .readyToPresentSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .renderFinishedTimeline = {make_unique<vk::raii::Semaphore>(*ctx.device, timelineSemaphoreInfo)},
            .computeFinishedTimeline = {make_unique<vk::raii::Semaphore>(*ctx.device, timelineSemaphoreInfo)},
        };
    }
}

// ==================== gui ====================

void AutomatonRenderer::initImgui() {
    const std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}
    };

    const vk::DescriptorPoolCreateInfo poolInfo = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1000,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    imguiDescriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);

    const std::uint32_t imageCount = SwapChain::getImageCount(ctx, *surface);

    ImGui_ImplVulkan_InitInfo imguiInitInfo = {
        .Instance = **instance,
        .PhysicalDevice = **ctx.physicalDevice,
        .Device = **ctx.device,
        .Queue = **graphicsQueue,
        .DescriptorPool = static_cast<VkDescriptorPool>(**imguiDescriptorPool),
        .MinImageCount = imageCount,
        .ImageCount = imageCount,
        .MSAASamples = static_cast<VkSampleCountFlagBits>(msaaSampleCount),
    };

    guiRenderer = make_unique<GuiRenderer>(window, imguiInitInfo, *renderPass);
}

void AutomatonRenderer::renderGuiSection() {
    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Renderer ", sectionFlags)) {
        const std::vector<std::tuple<std::string, ColoringPreset, std::function<void()> > > coloringPresets{
            {
                "Coord-based RGB", ColoringPreset::COORD_RGB, [&] {
                }
            },
            {
                "State-based gradient", ColoringPreset::STATE_GRADIENT, [&] {
                    ImGui::ColorEdit3("Color 1", &cellColor1.r);
                    ImGui::ColorEdit3("Color 2", &cellColor2.r);
                }
            },
            {
                "Center distance-based gradient", ColoringPreset::DISTANCE_GRADIENT, [&] {
                    ImGui::ColorEdit3("Color 1", &cellColor1.r);
                    ImGui::ColorEdit3("Color 2", &cellColor2.r);
                }
            },
            {
                "Solid color", ColoringPreset::SOLID_COLOR, [&] {
                    ImGui::ColorEdit3("Color", &cellColor1.r);
                }
            },
        };

        static int selectedPresetIdx = 0;
        constexpr auto comboFlags = ImGuiComboFlags_WidthFitPreview;

        ImGui::Text("Coloring preset:");
        if (ImGui::BeginCombo("##coloring_preset", std::get<0>(coloringPresets[selectedPresetIdx]).c_str(),
                              comboFlags)) {
            for (int i = 0; i < coloringPresets.size(); i++) {
                const bool isSelected = selectedPresetIdx == i;

                if (ImGui::Selectable(std::get<0>(coloringPresets[i]).c_str(), isSelected)) {
                    coloringPreset = std::get<1>(coloringPresets[i]);
                    selectedPresetIdx = i;
                }

                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        std::get<2>(coloringPresets[selectedPresetIdx])();

        ImGui::Separator();

        ImGui::DragFloat("Fog distance", &fogDistance, 0.1f, 0.0f, std::numeric_limits<float>::max(), "%.1f");

        ImGui::Separator();

        ImGui::Checkbox("Do neighbor shading?", &doNeighborShading);

        ImGui::Separator();

        ImGui::ColorEdit3("Background color", &backgroundColor.r);

        ImGui::Separator();

        ImGui::Checkbox("Use pyramid acceleration?", &usePyramidAcceleration);
    }

    camera->renderGuiSection();
}

// ==================== render loop ====================

void AutomatonRenderer::tick(const float deltaTime) {
    glfwPollEvents();
    camera->tick(deltaTime);
}

void AutomatonRenderer::renderGui(const std::function<void()> &renderCommands) const {
    const auto &commandBuffer = *frameResources[currentFrameIdx].guiCmdBuf;

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .renderPass = **renderPass,
        .framebuffer = *swapChain->getCurrentFramebuffer(),
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    if (doShowGui) {
        guiRenderer->startRendering();
        renderCommands();
        guiRenderer->finishRendering(commandBuffer);
    }

    commandBuffer.end();
}

void AutomatonRenderer::startFrame() {
    const auto &sync = frameResources[currentFrameIdx].sync;
    const auto &graphicsCmdBuf = frameResources[currentFrameIdx].graphicsCmdBuf;

    const std::vector waitSemaphores = {
        **sync.renderFinishedTimeline.semaphore,
    };

    const std::vector waitSemaphoreValues = {
        sync.renderFinishedTimeline.timeline,
    };

    const vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = static_cast<std::uint32_t>(waitSemaphores.size()),
        .pSemaphores = waitSemaphores.data(),
        .pValues = waitSemaphoreValues.data(),
    };

    if (ctx.device->waitSemaphores(waitInfo, UINT64_MAX) != vk::Result::eSuccess) {
        std::cerr << "waitSemaphores on renderFinishedTimeline failed" << std::endl;
    }

    updateGraphicsUniformBuffer();

    const auto &[result, imageIndex] = swapChain->acquireNextImage(*sync.imageAvailableSemaphore);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    graphicsCmdBuf->reset();
}

void AutomatonRenderer::endFrame() {
    auto &sync = frameResources[currentFrameIdx].sync;
    const auto &graphicsCmdBuf = frameResources[currentFrameIdx].graphicsCmdBuf;

    const std::vector waitSemaphores = {
        **sync.computeFinishedTimeline.semaphore,
        **sync.imageAvailableSemaphore
    };

    const std::vector waitSemaphoreValues = {
        sync.computeFinishedTimeline.timeline,
        static_cast<std::uint64_t>(0)
    };

    static constexpr vk::PipelineStageFlags waitStages[] = {
        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        vk::PipelineStageFlagBits::eVertexInput,
    };

    const std::array signalSemaphores = {
        **sync.renderFinishedTimeline.semaphore,
        **sync.readyToPresentSemaphore
    };

    sync.renderFinishedTimeline.timeline++;
    const std::vector signalSemaphoreValues{
        sync.renderFinishedTimeline.timeline,
        static_cast<std::uint64_t>(0)
    };

    const vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo{
        .waitSemaphoreValueCount = static_cast<std::uint32_t>(waitSemaphoreValues.size()),
        .pWaitSemaphoreValues = waitSemaphoreValues.data(),
        .signalSemaphoreValueCount = static_cast<std::uint32_t>(signalSemaphoreValues.size()),
        .pSignalSemaphoreValues = signalSemaphoreValues.data(),
    };

    const vk::SubmitInfo graphicsSubmitInfo{
        .pNext = &timelineSubmitInfo,
        .waitSemaphoreCount = static_cast<std::uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1U,
        .pCommandBuffers = &**graphicsCmdBuf,
        .signalSemaphoreCount = signalSemaphores.size(),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    graphicsQueue->submit(graphicsSubmitInfo);

    const std::array presentWaitSemaphores = {**sync.readyToPresentSemaphore};

    const std::array imageIndices = {swapChain->getCurrentImageIndex()};

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = presentWaitSemaphores.size(),
        .pWaitSemaphores = presentWaitSemaphores.data(),
        .swapchainCount = 1,
        .pSwapchains = &*swapChain->get(),
        .pImageIndices = imageIndices.data(),
    };

    auto presentResult = vk::Result::eSuccess;

    try {
        presentResult = presentQueue->presentKHR(presentInfo);
    } catch (...) {
    }

    const bool didResize = presentResult == vk::Result::eErrorOutOfDateKHR
                           || presentResult == vk::Result::eSuboptimalKHR
                           || framebufferResized;
    if (didResize) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (presentResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrameIdx = (currentFrameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
}

void AutomatonRenderer::drawScene() {
    recordGraphicsCommandBuffer();
}

void AutomatonRenderer::runCompute() {
    mostRecentSsboIdx = (mostRecentSsboIdx + 1) % AUTOMATON_RESOURCE_COUNT;

    auto &sync = frameResources[currentFrameIdx].sync;
    const auto &computeCmdBuf = frameResources[currentFrameIdx].computeCmdBuf;

    const std::vector waitSemaphores = {
        **sync.computeFinishedTimeline.semaphore,
    };

    const std::vector waitSemaphoreValues = {
        sync.computeFinishedTimeline.timeline,
    };

    const vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = static_cast<std::uint32_t>(waitSemaphores.size()),
        .pSemaphores = waitSemaphores.data(),
        .pValues = waitSemaphoreValues.data(),
    };

    if (ctx.device->waitSemaphores(waitInfo, UINT64_MAX) != vk::Result::eSuccess) {
        std::cerr << "waitSemaphores on computeFinishedTimeline failed" << std::endl;
    }

    computeCmdBuf->reset();
    recordComputeCommandBuffer();

    const std::array signalSemaphores = {**sync.computeFinishedTimeline.semaphore};

    sync.computeFinishedTimeline.timeline++;
    const std::vector signalSemaphoreValues{sync.computeFinishedTimeline.timeline};

    const vk::TimelineSemaphoreSubmitInfo timelineSubmitInfo{
        .signalSemaphoreValueCount = static_cast<std::uint32_t>(signalSemaphoreValues.size()),
        .pSignalSemaphoreValues = signalSemaphoreValues.data()
    };

    const vk::SubmitInfo computeSubmitInfo{
        .pNext = &timelineSubmitInfo,
        .commandBufferCount = 1U,
        .pCommandBuffers = &**computeCmdBuf,
        .signalSemaphoreCount = static_cast<std::uint32_t>(signalSemaphores.size()),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    computeQueue->submit(computeSubmitInfo);
}

void AutomatonRenderer::updateGraphicsUniformBuffer() const {
    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 proj = camera->getProjectionMatrix();

    glm::vec<2, int> windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    const GraphicsUBO graphicsUbo{
        .window = {
            .windowWidth = static_cast<std::uint32_t>(windowSize.x),
            .windowHeight = static_cast<std::uint32_t>(windowSize.y),
        },
        .matrices = {
            .view = view,
            .proj = proj,
            .inverseVp = glm::inverse(proj * view),
        },
        .coloring = {
            .doNeighborShading = doNeighborShading ? 1u : 0u,
            .coloringPreset = coloringPreset,
            .color1 = cellColor1,
            .color2 = cellColor2,
            .backgroundColor = backgroundColor
        },
        .misc = {
            .fogDistance = fogDistance,
            .cameraPos = camera->getPos(),
        },
        .automaton = {
            .gridDepth = automatonConfig.gridDepth,
            .stateCount = automatonConfig.preset.stateCount,
        },
    };

    memcpy(frameResources[currentFrameIdx].graphicsUboMapped, &graphicsUbo, sizeof(graphicsUbo));
}

void AutomatonRenderer::updateComputeUniformBuffers() const {
    const ComputeUBO computeUbo{
        .config = automatonConfig
    };

    for (size_t i = 0; i < AUTOMATON_RESOURCE_COUNT; i++) {
        memcpy(automatonResources[i].computeUboMapped, &computeUbo, sizeof(computeUbo));
    }
}
