#include "renderer.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
#include <ranges>

#include "gui/gui.h"
#include "mesh/model.h"
#include "mesh/vertex.h"
#include "vk/buffer.h"
#include "vk/swapchain.h"
#include "camera.h"

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

// vertices of a the skybox cube.
// might change this to be generated more intelligently... but it's good enough for now
static std::vector<SkyboxVertex> skyboxVertices = {
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},

    {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},

    {{-1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}},

    {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}}
};

VulkanRenderer::VulkanRenderer() {
    constexpr int INIT_WINDOW_WIDTH = 1200;
    constexpr int INIT_WINDOW_HEIGHT = 800;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "PBR Model Viewer", nullptr, nullptr);
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

    createScenePipeline();
    createSkyboxPipeline();

    createCommandPool();

    swapChain->createFramebuffers(ctx, *renderPass);

    createDescriptorPool();

    createUniformBuffers();

    createSkyboxResources();

    loadModel(
        "../assets/t-60-helmet/source/T-60 HelmetU.fbx",
        "../assets/t-60-helmet/textures/T-60_Helmet_DefaultMaterial_BaseColor.1001.png",
        "../assets/t-60-helmet/textures/T-60_Helmet_DefaultMaterial_Normal.1001.png",
        "../assets/t-60-helmet/textures/T-60_Helmet_DefaultMaterial_Roughness.1001.png"
    );

    createCommandBuffers();

    createSyncObjects();

    initImgui();
}

VulkanRenderer::~VulkanRenderer() {
    glfwDestroyWindow(window);
}

void VulkanRenderer::setIsCursorLocked(const bool b) const {
    camera->setIsCursorLocked(b);
    glfwSetInputMode(window, GLFW_CURSOR, b ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void VulkanRenderer::framebufferResizeCallback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto app = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

// ==================== instance creation ====================

void VulkanRenderer::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "PBR Model Viewer",
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

std::vector<const char *> VulkanRenderer::getRequiredExtensions() {
    std::uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// ==================== validation layers ====================

bool VulkanRenderer::checkValidationLayerSupport() {
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

vk::DebugUtilsMessengerCreateInfoEXT VulkanRenderer::makeDebugMessengerCreateInfo() {
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

void VulkanRenderer::setupDebugMessenger() {
    if constexpr (!enableValidationLayers) return;

    const vk::DebugUtilsMessengerCreateInfoEXT createInfo = makeDebugMessengerCreateInfo();
    debugMessenger = make_unique<vk::raii::DebugUtilsMessengerEXT>(*instance, createInfo);
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::debugCallback(
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

void VulkanRenderer::createSurface() {
    VkSurfaceKHR _surface;

    if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    surface = make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
}

// ==================== physical device ====================

void VulkanRenderer::pickPhysicalDevice() {
    const std::vector<vk::raii::PhysicalDevice> devices = instance->enumeratePhysicalDevices();

    for (const auto &dev: devices) {
        if (isDeviceSuitable(dev)) {
            ctx.physicalDevice = make_unique<vk::raii::PhysicalDevice>(dev);
            msaaSampleCount = getMaxUsableSampleCount();
            return;
        }
    }

    throw std::runtime_error("failed to find a suitable GPU!");
}

[[nodiscard]]
bool VulkanRenderer::isDeviceSuitable(const vk::raii::PhysicalDevice &physicalDevice) const {
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
QueueFamilyIndices VulkanRenderer::findQueueFamilies(const vk::raii::PhysicalDevice &physicalDevice) const {
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

bool VulkanRenderer::checkDeviceExtensionSupport(const vk::raii::PhysicalDevice &physicalDevice) {
    const std::vector<vk::ExtensionProperties> availableExtensions =
            physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto &extension: availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

bool VulkanRenderer::checkDeviceSubgroupSupport(const vk::raii::PhysicalDevice &physicalDevice) {
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

void VulkanRenderer::createLogicalDevice() {
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
    presentQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(presentFamily.value(), 0));
}

// ==================== models ====================

void VulkanRenderer::loadModel(const std::filesystem::path &meshPath, const std::filesystem::path &albedoPath,
                               const std::filesystem::path &normalPath, const std::filesystem::path &ormPath) {
    waitIdle();

    model = make_unique<Model>(meshPath);

    // this is only relevant when loading a new mesh and rebuilding the descriptors
    albedoTexture.reset();
    normalTexture.reset();
    ormTexture.reset();

    vertexBuffer.reset();
    indexBuffer.reset();
    for (auto &res: frameResources) {
        res.sceneDescriptorSet.reset();
    }

    Texture albedoTexRaw = TextureBuilder()
            .fromPaths({albedoPath})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    albedoTexture = make_unique<Texture>(std::move(albedoTexRaw));

    Texture normalTexRaw = TextureBuilder()
            .fromPaths({normalPath})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    normalTexture = make_unique<Texture>(std::move(normalTexRaw));

    Texture ormTexRaw = TextureBuilder()
            .fromPaths({ormPath})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    ormTexture = make_unique<Texture>(std::move(ormTexRaw));

    createVertexBuffer();
    createIndexBuffer();
    createSceneDescriptorSets();
}

void VulkanRenderer::createSkyboxTexture() {
    std::vector<std::filesystem::path> cubemapPaths{6, "../assets/skyboxes"};
    cubemapPaths[0].append("right.jpg");
    cubemapPaths[1].append("left.jpg");
    cubemapPaths[2].append("top.jpg");
    cubemapPaths[3].append("bottom.jpg");
    cubemapPaths[4].append("front.jpg");
    cubemapPaths[5].append("back.jpg");

    Texture cubemap = TextureBuilder()
            .asCubemap()
            .fromPaths(cubemapPaths)
            .create(ctx, *commandPool, *graphicsQueue);

    skyboxTexture = make_unique<Texture>(std::move(cubemap));
}

// ==================== swapchain ====================

void VulkanRenderer::recreateSwapChain() {
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

void VulkanRenderer::createDescriptorSetLayouts() {
    createSceneDescriptorSetLayouts();
    createSkyboxDescriptorSetLayouts();
}

void VulkanRenderer::createSceneDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding albedoSamplerLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding normalSamplerLayoutBinding{
        .binding = 2U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding ormSamplerLayoutBinding{
        .binding = 3U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        albedoSamplerLayoutBinding,
        normalSamplerLayoutBinding,
        ormSamplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<std::uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    sceneGraphicsSetLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createSkyboxDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        samplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<std::uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    skyboxGraphicsSetLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createDescriptorPool() {
    static constexpr vk::DescriptorPoolSize uboPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2,
    };

    static constexpr vk::DescriptorPoolSize samplerPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT) * 4,
    };

    static constexpr std::array poolSizes = {
        uboPoolSize,
        samplerPoolSize
    };

    static constexpr vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);
}

void VulkanRenderer::createSceneDescriptorSets() {
    constexpr std::uint32_t setsCount = MAX_FRAMES_IN_FLIGHT;

    const std::vector setLayouts(setsCount, **sceneGraphicsSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        const vk::DescriptorBufferInfo uboBufferInfo{
            .buffer = frameResources[i].graphicsUniformBuffer->get(),
            .offset = 0U,
            .range = sizeof(GraphicsUBO),
        };

        const vk::WriteDescriptorSet uboDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 0U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uboBufferInfo
        };

        const vk::DescriptorImageInfo albedoImageInfo{
            .sampler = *albedoTexture->getSampler(),
            .imageView = *albedoTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet albedoSamplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 1U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &albedoImageInfo
        };

        const vk::DescriptorImageInfo normalImageInfo{
            .sampler = *normalTexture->getSampler(),
            .imageView = *normalTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet normalSamplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 2U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &normalImageInfo
        };

        const vk::DescriptorImageInfo ormImageInfo{
            .sampler = *ormTexture->getSampler(),
            .imageView = *ormTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet ormSamplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 3U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &ormImageInfo
        };

        const std::array descriptorWrites = {
            uboDescriptorWrite,
            albedoSamplerDescriptorWrite,
            normalSamplerDescriptorWrite,
            ormSamplerDescriptorWrite
        };

        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        frameResources[i].sceneDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[i]));
    }
}

void VulkanRenderer::createSkyboxDescriptorSets() {
    constexpr std::uint32_t setsCount = MAX_FRAMES_IN_FLIGHT;

    const std::vector setLayouts(setsCount, **skyboxGraphicsSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        const vk::DescriptorBufferInfo uboBufferInfo{
            .buffer = frameResources[i].graphicsUniformBuffer->get(),
            .offset = 0U,
            .range = sizeof(GraphicsUBO),
        };

        const vk::WriteDescriptorSet uboDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 0U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uboBufferInfo
        };

        const vk::DescriptorImageInfo imageInfo{
            .sampler = *skyboxTexture->getSampler(),
            .imageView = *skyboxTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet samplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 1U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imageInfo
        };

        const std::array descriptorWrites = {
            uboDescriptorWrite,
            samplerDescriptorWrite
        };

        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        frameResources[i].skyboxDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[i]));
    }
}

// ==================== graphics pipeline ====================

void VulkanRenderer::createRenderPass() {
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

void VulkanRenderer::createPipelines() {
    createScenePipeline();
    createSkyboxPipeline();
}

void VulkanRenderer::createScenePipeline() {
    vk::raii::ShaderModule vertShaderModule = createShaderModule("../shaders/shader-vert.spv");
    vk::raii::ShaderModule fragShaderModule = createShaderModule("../shaders/shader-frag.spv");

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

    const std::vector shaderStages{
        vertShaderStageInfo,
        fragShaderStageInfo
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
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSampleCount,
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
        **sceneGraphicsSetLayout,
    };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = descriptorSetLayouts.size(),
        .pSetLayouts = descriptorSetLayouts.data(),
    };

    scenePipelineLayout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = shaderStages.size(),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = **scenePipelineLayout,
        .renderPass = **renderPass,
        .subpass = 0,
    };

    scenePipeline = make_unique<vk::raii::Pipeline>(*ctx.device, nullptr, pipelineInfo);
}

void VulkanRenderer::createSkyboxPipeline() {
    vk::raii::ShaderModule vertShaderModule = createShaderModule("../shaders/skybox-vert.spv");
    vk::raii::ShaderModule fragShaderModule = createShaderModule("../shaders/skybox-frag.spv");

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

    const std::vector shaderStages{
        vertShaderStageInfo,
        fragShaderStageInfo
    };

    const auto bindingDescription = SkyboxVertex::getBindingDescription();
    const auto attributeDescriptions = SkyboxVertex::getAttributeDescriptions();

    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
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
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSampleCount,
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
        .depthTestEnable = vk::False,
        .depthWriteEnable = vk::False,
    };

    const std::array descriptorSetLayouts = {
        **skyboxGraphicsSetLayout,
    };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = descriptorSetLayouts.size(),
        .pSetLayouts = descriptorSetLayouts.data(),
    };

    skyboxPipelineLayout = make_unique<vk::raii::PipelineLayout>(*ctx.device, pipelineLayoutInfo);

    const vk::GraphicsPipelineCreateInfo pipelineInfo{
        .stageCount = shaderStages.size(),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = **skyboxPipelineLayout,
        .renderPass = **renderPass,
        .subpass = 0,
    };

    skyboxPipeline = make_unique<vk::raii::Pipeline>(*ctx.device, nullptr, pipelineInfo);
}

[[nodiscard]]
vk::raii::ShaderModule VulkanRenderer::createShaderModule(const std::filesystem::path &path) const {
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
vk::SampleCountFlagBits VulkanRenderer::getMaxUsableSampleCount() const {
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

void VulkanRenderer::createSkyboxResources() {
    skyboxVertexBuffer = createLocalBuffer<SkyboxVertex>(skyboxVertices, vk::BufferUsageFlagBits::eVertexBuffer);
    createSkyboxTexture();
    createSkyboxDescriptorSets();
}

// ==================== buffers ====================

void VulkanRenderer::createVertexBuffer() {
    vertexBuffer = createLocalBuffer(model->getVertices(), vk::BufferUsageFlagBits::eVertexBuffer);
}

void VulkanRenderer::createIndexBuffer() {
    indexBuffer = createLocalBuffer(model->getIndices(), vk::BufferUsageFlagBits::eIndexBuffer);
}

template<typename ElemType>
unique_ptr<Buffer>
VulkanRenderer::createLocalBuffer(const std::vector<ElemType>& contents, const vk::BufferUsageFlags usage) {
    const vk::DeviceSize bufferSize = sizeof(contents[0]) * contents.size();

    Buffer stagingBuffer{
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();
    memcpy(data, contents.data(), static_cast<size_t>(bufferSize));
    stagingBuffer.unmap();

    auto resultBuffer = std::make_unique<Buffer>(
        ctx.allocator->get(),
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | usage,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    resultBuffer->copyFromBuffer(ctx, *commandPool, *graphicsQueue, stagingBuffer, bufferSize);

    return resultBuffer;
}

void VulkanRenderer::createUniformBuffers() {
    for (auto &res: frameResources) {
        res.graphicsUniformBuffer = std::make_unique<Buffer>(
            ctx.allocator->get(),
            sizeof(GraphicsUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        res.graphicsUboMapped = res.graphicsUniformBuffer->map();
    }
}

// ==================== commands ====================

void VulkanRenderer::createCommandPool() {
    const QueueFamilyIndices queueFamilyIndices = findQueueFamilies(*ctx.physicalDevice);

    const vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.graphicsComputeFamily.value()
    };

    commandPool = make_unique<vk::raii::CommandPool>(*ctx.device, poolInfo);
}

void VulkanRenderer::createCommandBuffers() {
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
    }
}

void VulkanRenderer::recordGraphicsCommandBuffer() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].graphicsCmdBuf;

    const vk::ClearColorValue clearColor{backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f};

    const std::array<vk::ClearValue, 2> clearValues{
        {
            clearColor,
            vk::ClearDepthStencilValue{
                .depth = 1.0f,
                .stencil = 0,
            }
        }
    };

    const vk::Extent2D swapChainExtent = swapChain->getExtent();

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = static_cast<float>(swapChainExtent.height), // flip the y-axis
        .width = static_cast<float>(swapChainExtent.width),
        .height = -1 * static_cast<float>(swapChainExtent.height), // flip the y-axis
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainExtent
    };

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

    constexpr vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInlineAndSecondaryCommandBuffersEXT);

    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    // skybox

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **skyboxPipeline);

    commandBuffer.bindVertexBuffers(0, skyboxVertexBuffer->get(), {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        **skyboxPipelineLayout,
        0,
        {
            **frameResources[currentFrameIdx].skyboxDescriptorSet,
        },
        nullptr
    );

    commandBuffer.draw(static_cast<std::uint32_t>(skyboxVertices.size()), 1, 0, 0);

    // scene

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **scenePipeline);

    commandBuffer.bindVertexBuffers(0, vertexBuffer->get(), {0});
    commandBuffer.bindIndexBuffer(indexBuffer->get(), 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        **scenePipelineLayout,
        0,
        {
            **frameResources[currentFrameIdx].sceneDescriptorSet,
        },
        nullptr
    );

    std::uint32_t indexOffset = 0;
    std::int32_t vertexOffset = 0;
    for (const auto& mesh : model->getMeshes()) {
        commandBuffer.drawIndexed(static_cast<std::uint32_t>(mesh.indices.size()), 1, indexOffset, vertexOffset, 0);
        indexOffset += static_cast<std::uint32_t>(mesh.indices.size());
        vertexOffset += static_cast<std::int32_t>(mesh.vertices.size());
    }

    commandBuffer.executeCommands(**frameResources[currentFrameIdx].guiCmdBuf);

    commandBuffer.endRenderPass();

    commandBuffer.end();
}

// ==================== sync ====================

void VulkanRenderer::createSyncObjects() {
    static constexpr vk::SemaphoreTypeCreateInfo typeCreateInfo{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };

    constexpr vk::SemaphoreCreateInfo timelineSemaphoreInfo{
        .pNext = &typeCreateInfo
    };

    constexpr vk::SemaphoreCreateInfo binarySemaphoreInfo;

    for (auto &res: frameResources) {
        res.sync = {
            .imageAvailableSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .readyToPresentSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .renderFinishedTimeline = {make_unique<vk::raii::Semaphore>(*ctx.device, timelineSemaphoreInfo)},
        };
    }
}

// ==================== gui ====================

void VulkanRenderer::initImgui() {
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

void VulkanRenderer::renderGuiSection() {
    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    // if (ImGui::CollapsingHeader("Renderer ", sectionFlags)) {
    //     ImGui::ColorEdit3("Background color", &backgroundColor.r);
    // }

    camera->renderGuiSection();
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float deltaTime) {
    glfwPollEvents();
    camera->tick(deltaTime);
}

void VulkanRenderer::renderGui(const std::function<void()> &renderCommands) const {
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

void VulkanRenderer::startFrame() {
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

void VulkanRenderer::endFrame() {
    auto &sync = frameResources[currentFrameIdx].sync;
    const auto &graphicsCmdBuf = frameResources[currentFrameIdx].graphicsCmdBuf;

    const std::vector waitSemaphores = {
        **sync.imageAvailableSemaphore
    };

    const std::vector waitSemaphoreValues = {
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

void VulkanRenderer::drawScene() {
    recordGraphicsCommandBuffer();
}

void VulkanRenderer::updateGraphicsUniformBuffer() const {
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
            .staticView = camera->getStaticViewMatrix(),
        },
        .misc = {
            .cameraPos = camera->getPos(),
            .lightDir = glm::normalize(glm::vec3(-1, 2, -3))
        }
    };

    memcpy(frameResources[currentFrameIdx].graphicsUboMapped, &graphicsUbo, sizeof(graphicsUbo));
}
