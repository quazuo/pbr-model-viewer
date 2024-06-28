#include "renderer.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <optional>
#include <set>
#include <vector>
#include <filesystem>
#include <array>

#include "gui/gui.h"
#include "mesh/model.h"
#include "mesh/vertex.h"
#include "vk/buffer.h"
#include "vk/swapchain.h"
#include "camera.h"
#include "vk/cmd.h"
#include "src/utils/glfw-statics.h"
#include "vk/pipeline.h"

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

VulkanRenderer::VulkanRenderer() {
    constexpr int INIT_WINDOW_WIDTH = 1200;
    constexpr int INIT_WINDOW_HEIGHT = 800;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(INIT_WINDOW_WIDTH, INIT_WINDOW_HEIGHT, "PBR Model Viewer", nullptr, nullptr);

    initGlfwUserPointer(window);
    auto *userData = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!userData) throw std::runtime_error("unexpected null window user pointer");
    userData->renderer = this;

    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    camera = make_unique<Camera>(window);

    inputManager = make_unique<InputManager>(window);
    bindMouseDragActions();

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
    createCommandBuffers();

    swapChain->createFramebuffers(ctx, *sceneRenderPass);

    createDescriptorPool();

    createUniformBuffers();
    updateGraphicsUniformBuffer();

    createSkyboxVertexBuffer();
    createScreenSpaceQuadVertexBuffer();

    createCubemapCaptureRenderPass();
    createCubemapCapturePipeline();

    createCubemapConvoluteRenderPass();
    createIrradianceCapturePipeline();

    createPrefilterPipeline();

    createBrdfIntegrationRenderPass();
    createBrdfIntegrationPipeline();

    // loadModel("../assets/t-60-helmet/source/T-60 HelmetU.fbx");
    // loadAlbedoTexture("../assets/t-60-helmet/textures/albedo.png");
    // loadNormalMap("../assets/t-60-helmet/textures/normal.png");
    // loadOrmMap("../assets/t-60-helmet/textures/orm.png");

    loadModel("../assets/default-model/czajnik.obj");
    loadAlbedoTexture("../assets/default-model/czajnik-albedo.png");
    loadNormalMap("../assets/default-model/czajnik-normal.png");
    loadOrmMap("../assets/default-model/czajnik-orm.png");

    loadEnvironmentMap("../assets/envmaps/gallery.hdr");

    createBrdfIntegrationMapTexture();
    computeBrdfIntegrationMap();

    createSyncObjects();

    initImgui();
}

VulkanRenderer::~VulkanRenderer() {
    glfwDestroyWindow(window);
}

void VulkanRenderer::framebufferResizeCallback(GLFWwindow *window, const int width, const int height) {
    (void) (width + height);
    const auto userData = static_cast<GlfwStaticUserData *>(glfwGetWindowUserPointer(window));
    if (!userData) throw std::runtime_error("unexpected null window user pointer");
    userData->renderer->framebufferResized = true;
}

void VulkanRenderer::bindMouseDragActions() {
    inputManager->bindMouseDragCallback(GLFW_MOUSE_BUTTON_RIGHT, [&](const double dx, const double dy) {
        static constexpr float speed = 0.002;
        const float cameraDistance = glm::length(camera->getPos());
        const float inverseScale = 1.0f / (modelScale + 0.001f);

        const auto viewVectors = camera->getViewVectors();

        modelTranslate += cameraDistance * inverseScale * speed * viewVectors.right * static_cast<float>(dx);
        modelTranslate -= cameraDistance * inverseScale * speed * viewVectors.up * static_cast<float>(dy);
    });
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
        .enabledLayerCount = static_cast<uint32_t>(enableValidationLayers ? validationLayers.size() : 0),
        .ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    instance = make_unique<vk::raii::Instance>(vkCtx, createInfo);
}

std::vector<const char *> VulkanRenderer::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

// ==================== validation layers ====================

bool VulkanRenderer::checkValidationLayerSupport() {
    uint32_t layerCount;
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

    return true;
}

QueueFamilyIndices VulkanRenderer::findQueueFamilies(const vk::raii::PhysicalDevice &physicalDevice) const {
    const std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

    std::optional<uint32_t> graphicsComputeFamily;
    std::optional<uint32_t> presentFamily;

    uint32_t i = 0;
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

// ==================== logical device ====================

void VulkanRenderer::createLogicalDevice() {
    const auto [graphicsComputeFamily, presentFamily] = findQueueFamilies(*ctx.physicalDevice);
    const std::set uniqueQueueFamilies = {graphicsComputeFamily.value(), presentFamily.value()};

    constexpr float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    for (uint32_t queueFamily: uniqueQueueFamilies) {
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
        .timelineSemaphore = vk::True,
    };

    const vk::DeviceCreateInfo createInfo{
        .pNext = &vulkan12Features,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(enableValidationLayers ? validationLayers.size() : 0),
        .ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures,
    };

    ctx.device = make_unique<vk::raii::Device>(*ctx.physicalDevice, createInfo);

    graphicsQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(graphicsComputeFamily.value(), 0));
    presentQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(presentFamily.value(), 0));
}

// ==================== models ====================

void VulkanRenderer::loadModel(const std::filesystem::path &path) {
    waitIdle();

    model = make_unique<Model>(path);

    vertexBuffer.reset();
    indexBuffer.reset();

    createVertexBuffer();
    createIndexBuffer();
}

void VulkanRenderer::loadAlbedoTexture(const std::filesystem::path &path) {
    waitIdle();

    albedoTexture.reset();

    albedoTexture = TextureBuilder()
            .fromPaths({path})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);
}

void VulkanRenderer::loadNormalMap(const std::filesystem::path &path) {
    waitIdle();

    normalTexture.reset();

    normalTexture = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx, *commandPool, *graphicsQueue);
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &path) {
    waitIdle();

    ormTexture.reset();

    ormTexture = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx, *commandPool, *graphicsQueue);
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &aoPath, const std::filesystem::path &roughnessPath,
                                const std::filesystem::path &metallicPath) {
    waitIdle();

    ormTexture.reset();

    ormTexture = TextureBuilder()
            .asSeparateChannels()
            .fromPaths({aoPath, roughnessPath, metallicPath})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);
}

void VulkanRenderer::loadEnvironmentMap(const std::filesystem::path &path) {
    waitIdle();

    for (auto &res: frameResources) {
        res.sceneDescriptorSet.reset();
        res.skyboxDescriptorSet.reset();
    }

    cubemapCaptureDescriptorSet.reset();
    envmapConvoluteDescriptorSet.reset();

    createEnvmapTextures(path);

    createSceneDescriptorSets();
    createSkyboxDescriptorSets();
    createCubemapCaptureDescriptorSets();
    createEnvmapConvoluteDescriptorSets();

    createCubemapCaptureFramebuffer();
    createIrradianceCaptureFramebuffer();
    createPrefilterFramebuffers();

    captureCubemap();
    utils::img::transitionImageLayout(
        ctx,
        **skyboxTexture->getImage(),
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        skyboxTexture->getMipLevels(),
        6,
        *commandPool,
        *graphicsQueue
    );
    skyboxTexture->generateMipmaps(ctx, *commandPool, *graphicsQueue, vk::ImageLayout::eShaderReadOnlyOptimal);

    captureIrradianceMap();
    utils::img::transitionImageLayout(
        ctx,
        **irradianceMapTexture->getImage(),
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        irradianceMapTexture->getMipLevels(),
        6,
        *commandPool,
        *graphicsQueue
    );
    irradianceMapTexture->generateMipmaps(ctx, *commandPool, *graphicsQueue, vk::ImageLayout::eShaderReadOnlyOptimal);

    prefilterEnvmap();
}

void VulkanRenderer::buildDescriptors() {
    for (auto &res: frameResources) {
        res.sceneDescriptorSet.reset();
    }

    createSceneDescriptorSets();
}

void VulkanRenderer::createEnvmapTextures(const std::filesystem::path &path) {
    skyboxTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({cubemapExtent.width, cubemapExtent.height, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    envmapTexture = TextureBuilder()
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .fromPaths({path})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    irradianceMapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({32, 32, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    prefilteredEnvmapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({128, 128, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);
}

void VulkanRenderer::createBrdfIntegrationMapTexture() {
    brdfIntegrationMapTexture = TextureBuilder()
            .asUninitialized({brdfIntegrationMapExtent.width, brdfIntegrationMapExtent.height, 1})
            .useFormat(brdfIntegrationMapFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);
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
    swapChain = make_unique<SwapChain>(
        ctx,
        *surface,
        findQueueFamilies(*ctx.physicalDevice),
        window,
        msaaSampleCount
    );
    swapChain->createFramebuffers(ctx, *sceneRenderPass);
}

// ==================== descriptors ====================

void VulkanRenderer::createDescriptorSetLayouts() {
    createSceneDescriptorSetLayouts();
    createSkyboxDescriptorSetLayouts();
    createCubemapCaptureDescriptorSetLayouts();
    createEnvmapConvoluteDescriptorSetLayouts();
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

    static constexpr vk::DescriptorSetLayoutBinding irradianceMapSamplerLayoutBinding{
        .binding = 4U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        albedoSamplerLayoutBinding,
        normalSamplerLayoutBinding,
        ormSamplerLayoutBinding,
        irradianceMapSamplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    sceneDescriptorLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createSkyboxDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr vk::DescriptorSetLayoutBinding cubemapSamplerLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        cubemapSamplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    skyboxDescriptorLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createCubemapCaptureDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };

    static constexpr vk::DescriptorSetLayoutBinding envmapSamplerLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        envmapSamplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    cubemapCaptureDescriptorLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createEnvmapConvoluteDescriptorSetLayouts() {
    static constexpr vk::DescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    };

    static constexpr vk::DescriptorSetLayoutBinding skyboxSamplerLayoutBinding{
        .binding = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1U,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };

    static constexpr std::array setBindings{
        uboLayoutBinding,
        skyboxSamplerLayoutBinding,
    };

    static constexpr vk::DescriptorSetLayoutCreateInfo setLayoutInfo{
        .bindingCount = static_cast<uint32_t>(setBindings.size()),
        .pBindings = setBindings.data(),
    };

    envmapConvoluteDescriptorLayout = make_unique<vk::raii::DescriptorSetLayout>(*ctx.device, setLayoutInfo);
}

void VulkanRenderer::createDescriptorPool() {
    static constexpr vk::DescriptorPoolSize uboPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2 + 3,
    };

    static constexpr vk::DescriptorPoolSize samplerPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 5 + 2,
    };

    static constexpr std::array poolSizes = {
        uboPoolSize,
        samplerPoolSize
    };

    static constexpr vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2 + 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);
}

void VulkanRenderer::createSceneDescriptorSets() {
    constexpr uint32_t setsCount = MAX_FRAMES_IN_FLIGHT;

    const std::vector setLayouts(setsCount, **sceneDescriptorLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        const vk::DescriptorBufferInfo uboBufferInfo{
            .buffer = **frameResources[i].graphicsUniformBuffer,
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

        const vk::DescriptorImageInfo irradianceMapImageInfo{
            .sampler = *irradianceMapTexture->getSampler(),
            .imageView = *irradianceMapTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet irradianceMapSamplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 4U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &irradianceMapImageInfo
        };

        const std::array descriptorWrites = {
            uboDescriptorWrite,
            albedoSamplerDescriptorWrite,
            normalSamplerDescriptorWrite,
            ormSamplerDescriptorWrite,
            irradianceMapSamplerDescriptorWrite
        };

        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        frameResources[i].sceneDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[i]));
    }
}

void VulkanRenderer::createSkyboxDescriptorSets() {
    constexpr uint32_t setsCount = MAX_FRAMES_IN_FLIGHT;

    const std::vector setLayouts(setsCount, **skyboxDescriptorLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = setsCount,
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < setsCount; i++) {
        const vk::DescriptorBufferInfo uboBufferInfo{
            .buffer = **frameResources[i].graphicsUniformBuffer,
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

        const vk::DescriptorImageInfo skyboxImageInfo{
            .sampler = *skyboxTexture->getSampler(),
            .imageView = *skyboxTexture->getView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet skyboxSamplerDescriptorWrite{
            .dstSet = *descriptorSets[i],
            .dstBinding = 1U,
            .dstArrayElement = 0U,
            .descriptorCount = 1U,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &skyboxImageInfo
        };

        const std::array descriptorWrites = {
            uboDescriptorWrite,
            skyboxSamplerDescriptorWrite,
        };

        ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

        frameResources[i].skyboxDescriptorSet =
                make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[i]));
    }
}

void VulkanRenderer::createCubemapCaptureDescriptorSets() {
    constexpr uint32_t setsCount = 1;
    const std::vector setLayouts(setsCount, **cubemapCaptureDescriptorLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    const vk::DescriptorBufferInfo uboBufferInfo{
        .buffer = **frameResources[0].graphicsUniformBuffer,
        .offset = 0U,
        .range = sizeof(GraphicsUBO),
    };

    const vk::WriteDescriptorSet uboDescriptorWrite{
        .dstSet = *descriptorSets[0],
        .dstBinding = 0U,
        .dstArrayElement = 0U,
        .descriptorCount = 1U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &uboBufferInfo
    };

    const vk::DescriptorImageInfo envmapImageInfo{
        .sampler = *envmapTexture->getSampler(),
        .imageView = *envmapTexture->getView(),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet envmapSamplerDescriptorWrite{
        .dstSet = *descriptorSets[0],
        .dstBinding = 1U,
        .dstArrayElement = 0U,
        .descriptorCount = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &envmapImageInfo
    };

    const std::array descriptorWrites = {
        uboDescriptorWrite,
        envmapSamplerDescriptorWrite
    };

    ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

    cubemapCaptureDescriptorSet = make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[0]));
}

void VulkanRenderer::createEnvmapConvoluteDescriptorSets() {
    constexpr uint32_t setsCount = 1;
    const std::vector setLayouts(setsCount, **envmapConvoluteDescriptorLayout);

    const vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = **descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
    };

    std::vector<vk::raii::DescriptorSet> descriptorSets = ctx.device->allocateDescriptorSets(allocInfo);

    const vk::DescriptorBufferInfo uboBufferInfo{
        .buffer = **frameResources[0].graphicsUniformBuffer,
        .offset = 0U,
        .range = sizeof(GraphicsUBO),
    };

    const vk::WriteDescriptorSet uboDescriptorWrite{
        .dstSet = *descriptorSets[0],
        .dstBinding = 0U,
        .dstArrayElement = 0U,
        .descriptorCount = 1U,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &uboBufferInfo
    };

    const vk::DescriptorImageInfo skyboxImageInfo{
        .sampler = *skyboxTexture->getSampler(),
        .imageView = *skyboxTexture->getView(),
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet skyboxSamplerDescriptorWrite{
        .dstSet = *descriptorSets[0],
        .dstBinding = 1U,
        .dstArrayElement = 0U,
        .descriptorCount = 1U,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &skyboxImageInfo
    };

    const std::array descriptorWrites = {
        uboDescriptorWrite,
        skyboxSamplerDescriptorWrite
    };

    ctx.device->updateDescriptorSets(descriptorWrites, nullptr);

    envmapConvoluteDescriptorSet = make_unique<vk::raii::DescriptorSet>(std::move(descriptorSets[0]));
}

// ==================== render passes ====================

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
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };

    sceneRenderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
}

void VulkanRenderer::createCubemapCaptureRenderPass() {
    static constexpr vk::AttachmentDescription colorAttachment{
        .format = hdrEnvmapFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    const std::vector attachments{6, colorAttachment};

    std::vector<vk::AttachmentReference> attachmentRefs;

    for (uint32_t i = 0; i < 6; i++) {
        const vk::AttachmentReference colorAttachmentRef{
            .attachment = i,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        attachmentRefs.push_back(colorAttachmentRef);
    }

    std::vector<vk::SubpassDescription> subpasses;
    std::vector<vk::SubpassDependency> dependencies;

    for (uint32_t i = 0; i < 6; i++) {
        const vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentRefs[i],
        };

        subpasses.push_back(subpass);

        const vk::SubpassDependency dependency{
            .srcSubpass = i == 0 ? vk::SubpassExternal : i - 1,
            .dstSubpass = i,
            .srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                             | vk::AccessFlagBits::eDepthStencilAttachmentWrite
        };

        dependencies.push_back(dependency);
    }

    const vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = static_cast<uint32_t>(subpasses.size()),
        .pSubpasses = subpasses.data(),
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data()
    };

    cubemapCaptureRenderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
}

void VulkanRenderer::createCubemapConvoluteRenderPass() {
    static constexpr vk::AttachmentDescription colorAttachment{
        .format = hdrEnvmapFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    const std::vector attachments{6, colorAttachment};

    std::vector<vk::AttachmentReference> attachmentRefs;

    for (uint32_t i = 0; i < 6; i++) {
        const vk::AttachmentReference colorAttachmentRef{
            .attachment = i,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        attachmentRefs.push_back(colorAttachmentRef);
    }

    std::vector<vk::SubpassDescription> subpasses;
    std::vector<vk::SubpassDependency> dependencies;

    for (uint32_t i = 0; i < 6; i++) {
        const vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentRefs[i],
        };

        subpasses.push_back(subpass);

        const vk::SubpassDependency dependency{
            .srcSubpass = i == 0 ? vk::SubpassExternal : i - 1,
            .dstSubpass = i,
            .srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                             | vk::AccessFlagBits::eDepthStencilAttachmentWrite
        };

        dependencies.push_back(dependency);
    }

    const vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = static_cast<uint32_t>(subpasses.size()),
        .pSubpasses = subpasses.data(),
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies = dependencies.data()
    };

    envmapConvoluteRenderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
}

void VulkanRenderer::createBrdfIntegrationRenderPass() {
    static constexpr vk::AttachmentDescription colorAttachment{
        .format = brdfIntegrationMapFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    static constexpr vk::AttachmentReference attachmentRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    static constexpr vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentRef,
    };

    static constexpr vk::SubpassDependency dependency{
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                        | vk::PipelineStageFlagBits::eLateFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests
                        | vk::PipelineStageFlagBits::eLateFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead
                         | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    };

    static constexpr vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };

    brdfIntegrationRenderPass = make_unique<vk::raii::RenderPass>(*ctx.device, renderPassInfo);
}

// ==================== pipelines ====================

void VulkanRenderer::createScenePipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/main-vert.spv")
            .withFragmentShader("../shaders/obj/main-frag.spv")
            .withVertices<Vertex>()
            .withMultisampling({
                .rasterizationSamples = msaaSampleCount,
                .minSampleShading = 1.0f,
            })
            .withDescriptorLayouts({
                **sceneDescriptorLayout,
            })
            .create(ctx, *sceneRenderPass);

    scenePipeline = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createSkyboxPipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/skybox-vert.spv")
            .withFragmentShader("../shaders/obj/skybox-frag.spv")
            .withVertices<SkyboxVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withMultisampling({
                .rasterizationSamples = msaaSampleCount,
                .minSampleShading = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .withDescriptorLayouts({
                **skyboxDescriptorLayout,
            })
            .create(ctx, *sceneRenderPass);

    skyboxPipeline = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createCubemapCapturePipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/sphere-cube-vert.spv")
            .withFragmentShader("../shaders/obj/sphere-cube-frag.spv")
            .withVertices<SkyboxVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .withDescriptorLayouts({
                **cubemapCaptureDescriptorLayout,
            })
            .withPushConstants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eVertex,
                    .offset = 0,
                    .size = sizeof(SkyboxPushConstants),
                }
            })
            .forSubpasses(6)
            .create(ctx, *cubemapCaptureRenderPass);

    cubemapCapturePipelines = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createIrradianceCapturePipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/convolute-vert.spv")
            .withFragmentShader("../shaders/obj/convolute-frag.spv")
            .withVertices<SkyboxVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .withDescriptorLayouts({
                **envmapConvoluteDescriptorLayout,
            })
            .withPushConstants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eVertex,
                    .offset = 0,
                    .size = sizeof(SkyboxPushConstants),
                }
            })
            .forSubpasses(6)
            .create(ctx, *envmapConvoluteRenderPass);

    irradianceCapturePipelines = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createPrefilterPipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/prefilter-vert.spv")
            .withFragmentShader("../shaders/obj/prefilter-frag.spv")
            .withVertices<SkyboxVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .withDescriptorLayouts({
                **envmapConvoluteDescriptorLayout,
            })
            .withPushConstants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                    .offset = 0,
                    .size = sizeof(PrefilterPushConstants),
                }
            })
            .forSubpasses(6)
            .create(ctx, *envmapConvoluteRenderPass);

    prefilterPipelines = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createBrdfIntegrationPipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/brdf-integrate-vert.spv")
            .withFragmentShader("../shaders/obj/brdf-integrate-frag.spv")
            .withVertices<ScreenSpaceQuadVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .create(ctx, *brdfIntegrationRenderPass);

    brdfIntegrationPipeline = make_unique<PipelinePack>(std::move(pipeline));
}

// ==================== multisampling ====================

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

// ==================== buffers ====================

void VulkanRenderer::createVertexBuffer() {
    vertexBuffer = createLocalBuffer(model->getVertices(), vk::BufferUsageFlagBits::eVertexBuffer);
    instanceDataBuffer = createLocalBuffer(model->getInstanceTransforms(), vk::BufferUsageFlagBits::eVertexBuffer);
}

void VulkanRenderer::createSkyboxVertexBuffer() {
    skyboxVertexBuffer = createLocalBuffer<SkyboxVertex>(skyboxVertices, vk::BufferUsageFlagBits::eVertexBuffer);
}

void VulkanRenderer::createScreenSpaceQuadVertexBuffer() {
    screenSpaceQuadVertexBuffer = createLocalBuffer<ScreenSpaceQuadVertex>(
        screenSpaceQuadVertices,
        vk::BufferUsageFlagBits::eVertexBuffer
    );
}

void VulkanRenderer::createIndexBuffer() {
    indexBuffer = createLocalBuffer(model->getIndices(), vk::BufferUsageFlagBits::eIndexBuffer);
}

template<typename ElemType>
unique_ptr<Buffer>
VulkanRenderer::createLocalBuffer(const std::vector<ElemType> &contents, const vk::BufferUsageFlags usage) {
    const vk::DeviceSize bufferSize = sizeof(contents[0]) * contents.size();

    Buffer stagingBuffer{
        **ctx.allocator,
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    void *data = stagingBuffer.map();
    memcpy(data, contents.data(), static_cast<size_t>(bufferSize));
    stagingBuffer.unmap();

    auto resultBuffer = make_unique<Buffer>(
        **ctx.allocator,
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst | usage,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    resultBuffer->copyFromBuffer(ctx, *commandPool, *graphicsQueue, stagingBuffer, bufferSize);

    return resultBuffer;
}

void VulkanRenderer::createUniformBuffers() {
    for (auto &res: frameResources) {
        res.graphicsUniformBuffer = make_unique<Buffer>(
            **ctx.allocator,
            sizeof(GraphicsUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        res.graphicsUboMapped = res.graphicsUniformBuffer->map();
    }
}

void VulkanRenderer::createCubemapCaptureFramebuffer() {
    cubemapCaptureFramebuffer = createPerLayerCubemapFramebuffer(
        *skyboxTexture,
        *cubemapCaptureRenderPass
    );
}

void VulkanRenderer::createIrradianceCaptureFramebuffer() {
    irradianceCaptureFramebuffer = createPerLayerCubemapFramebuffer(
        *irradianceMapTexture,
        *envmapConvoluteRenderPass
    );
}

void VulkanRenderer::createPrefilterFramebuffers() {
    for (uint32_t mip = 0; mip < maxPrefilterMipLevels; mip++) {
        prefilterFramebuffers.emplace_back(
            createMipPerLayerCubemapFramebuffer(
                *prefilteredEnvmapTexture,
                *envmapConvoluteRenderPass,
                mip
            )
        );
    }
}

unique_ptr<vk::raii::Framebuffer>
VulkanRenderer::createPerLayerCubemapFramebuffer(const Texture &texture, const vk::raii::RenderPass &renderPass) const {
    std::vector<vk::ImageView> attachments;

    for (uint32_t i = 0; i < 6; i++) {
        attachments.push_back(*texture.getAttachmentLayerView(i));
    }

    const vk::FramebufferCreateInfo createInfo{
        .renderPass = *renderPass,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = texture.getImage().getExtent().width,
        .height = texture.getImage().getExtent().height,
        .layers = 1,
    };

    return make_unique<vk::raii::Framebuffer>(*ctx.device, createInfo);
}

unique_ptr<vk::raii::Framebuffer>
VulkanRenderer::createMipPerLayerCubemapFramebuffer(const Texture &texture, const vk::raii::RenderPass &renderPass,
                                                    const uint32_t mipLevel) const {
    std::vector<vk::ImageView> attachments;

    for (uint32_t layer = 0; layer < 6; layer++) {
        attachments.push_back(*texture.getLayerMipView(layer, mipLevel));
    }

    const uint32_t mipScalingFactor = 1 << mipLevel;

    const vk::FramebufferCreateInfo createInfo{
        .renderPass = *renderPass,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = texture.getImage().getExtent().width / mipScalingFactor,
        .height = texture.getImage().getExtent().height / mipScalingFactor,
        .layers = 1,
    };

    return make_unique<vk::raii::Framebuffer>(*ctx.device, createInfo);
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
        .commandBufferCount = static_cast<uint32_t>(frameResources.size()),
    };

    const vk::CommandBufferAllocateInfo secondaryAllocInfo{
        .commandPool = **commandPool,
        .level = vk::CommandBufferLevel::eSecondary,
        .commandBufferCount = static_cast<uint32_t>(frameResources.size()),
    };

    vk::raii::CommandBuffers graphicsCommandBuffers{*ctx.device, primaryAllocInfo};
    vk::raii::CommandBuffers sceneCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers guiCommandBuffers{*ctx.device, secondaryAllocInfo};

    for (size_t i = 0; i < graphicsCommandBuffers.size(); i++) {
        frameResources[i].graphicsCmdBuffer =
                make_unique<vk::raii::CommandBuffer>(std::move(graphicsCommandBuffers[i]));
        frameResources[i].sceneCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(sceneCommandBuffers[i]))};
        frameResources[i].guiCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(guiCommandBuffers[i]))};
    }

    const vk::CommandBufferAllocateInfo cubemapCapturePrimaryAllocInfo{
        .commandPool = **commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1u,
    };

    vk::raii::CommandBuffers cubemapCaptureCommandBuffers{*ctx.device, cubemapCapturePrimaryAllocInfo};
}

void VulkanRenderer::recordGraphicsCommandBuffer() const {
    const auto &commandBuffer = *frameResources[currentFrameIdx].graphicsCmdBuffer;

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

    const vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = **sceneRenderPass,
        .framebuffer = *swapChain->getCurrentFramebuffer(),
        .renderArea = {
            .offset = {0, 0},
            .extent = swapChainExtent
        },
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
    };

    constexpr vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eSecondaryCommandBuffers);

    if (frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].sceneCmdBuffer.buffer);
    }

    if (frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].guiCmdBuffer.buffer);
    }

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
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    imguiDescriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);

    const uint32_t imageCount = SwapChain::getImageCount(ctx, *surface);

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

    guiRenderer = make_unique<GuiRenderer>(window, imguiInitInfo, *sceneRenderPass);
}

void VulkanRenderer::renderGuiSection() {
    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Renderer ", sectionFlags)) {
        ImGui::DragFloat("Model scale", &modelScale, 0.01, 0, std::numeric_limits<float>::max());

        ImGui::Separator();

        ImGui::Checkbox("Use IBL?", &useIBL);
    }

    camera->renderGuiSection();
}

// ==================== render loop ====================

void VulkanRenderer::tick(const float deltaTime) {
    glfwPollEvents();
    camera->tick(deltaTime);

    if (
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)
        && !ImGui::IsAnyItemActive()
        && !ImGui::IsAnyItemFocused()
    ) {
        inputManager->tick(deltaTime);
    }
}

void VulkanRenderer::renderGui(const std::function<void()> &renderCommands) {
    const auto &commandBuffer = *frameResources[currentFrameIdx].guiCmdBuffer.buffer;

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .renderPass = **sceneRenderPass,
        .framebuffer = *swapChain->getCurrentFramebuffer(),
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    guiRenderer->startRendering();
    renderCommands();
    guiRenderer->finishRendering(commandBuffer);

    commandBuffer.end();

    frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame = true;
}

bool VulkanRenderer::startFrame() {
    const auto &sync = frameResources[currentFrameIdx].sync;

    const std::vector waitSemaphores = {
        **sync.renderFinishedTimeline.semaphore,
    };

    const std::vector waitSemaphoreValues = {
        sync.renderFinishedTimeline.timeline,
    };

    const vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
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
        return false;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    frameResources[currentFrameIdx].graphicsCmdBuffer->reset();

    frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame = false;
    frameResources[currentFrameIdx].sceneCmdBuffer.buffer->reset();

    frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame = false;
    frameResources[currentFrameIdx].guiCmdBuffer.buffer->reset();

    return true;
}

void VulkanRenderer::endFrame() {
    recordGraphicsCommandBuffer();

    auto &sync = frameResources[currentFrameIdx].sync;

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
        .waitSemaphoreValueCount = static_cast<uint32_t>(waitSemaphoreValues.size()),
        .pWaitSemaphoreValues = waitSemaphoreValues.data(),
        .signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size()),
        .pSignalSemaphoreValues = signalSemaphoreValues.data(),
    };

    const vk::SubmitInfo graphicsSubmitInfo{
        .pNext = &timelineSubmitInfo,
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1U,
        .pCommandBuffers = &**frameResources[currentFrameIdx].graphicsCmdBuffer,
        .signalSemaphoreCount = signalSemaphores.size(),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    graphicsQueue->submit(graphicsSubmitInfo);

    const std::array presentWaitSemaphores = {**sync.readyToPresentSemaphore};

    const std::array imageIndices = {swapChain->getCurrentImageIndex()};

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = presentWaitSemaphores.size(),
        .pWaitSemaphores = presentWaitSemaphores.data(),
        .swapchainCount = 1U,
        .pSwapchains = &***swapChain,
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
    const auto &commandBuffer = *frameResources[currentFrameIdx].sceneCmdBuffer.buffer;

    const vk::Extent2D swapChainExtent = swapChain->getExtent();

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = static_cast<float>(swapChainExtent.height),
        .width = static_cast<float>(swapChainExtent.width),
        .height = -1 * static_cast<float>(swapChainExtent.height), // flip the y-axis
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainExtent
    };

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .renderPass = **sceneRenderPass,
        .framebuffer = *swapChain->getCurrentFramebuffer(),
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    // skybox

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***skyboxPipeline);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *skyboxPipeline->getLayout(),
        0,
        {
            **frameResources[currentFrameIdx].skyboxDescriptorSet,
        },
        nullptr
    );

    commandBuffer.draw(static_cast<uint32_t>(skyboxVertices.size()), 1, 0, 0);

    // scene

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***scenePipeline);

    commandBuffer.bindVertexBuffers(0, **vertexBuffer, {0});
    commandBuffer.bindVertexBuffers(1, **instanceDataBuffer, {0});

    commandBuffer.bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *scenePipeline->getLayout(),
        0,
        {
            **frameResources[currentFrameIdx].sceneDescriptorSet,
        },
        nullptr
    );

    if (model) {
        uint32_t indexOffset = 0;
        std::int32_t vertexOffset = 0;
        uint32_t instanceOffset = 0;

        for (const auto &mesh: model->getMeshes()) {
            commandBuffer.drawIndexed(
                static_cast<uint32_t>(mesh.indices.size()),
                static_cast<uint32_t>(mesh.instances.size()),
                indexOffset,
                vertexOffset,
                instanceOffset
            );

            indexOffset += static_cast<uint32_t>(mesh.indices.size());
            vertexOffset += static_cast<std::int32_t>(mesh.vertices.size());
            instanceOffset += static_cast<uint32_t>(mesh.instances.size());
        }
    }

    commandBuffer.end();

    frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame = true;
}

static const glm::mat4 cubemapFaceProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

static const std::array cubemapFaceViews{
    glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0)),
    glm::lookAt(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)),
    glm::lookAt(glm::vec3(0), glm::vec3(0, 1, 0), glm::vec3(0, 0, -1)),
    glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1)),
    glm::lookAt(glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(0, 1, 0)),
    glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0))
};

void VulkanRenderer::captureCubemap() {
    const vk::Extent2D extent = cubemapExtent;

    constexpr vk::ClearColorValue clearColor{0, 0, 0, 1};
    const std::vector<vk::ClearValue> clearValues{6, clearColor};

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = static_cast<float>(extent.height),
        .width = static_cast<float>(extent.width),
        .height = -1 * static_cast<float>(extent.height), // flip the y-axis
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = extent
    };

    const vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = **cubemapCaptureRenderPass,
        .framebuffer = **cubemapCaptureFramebuffer,
        .renderArea = {
            .offset = {0, 0},
            .extent = extent,
        },
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
    };

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *cubemapCapturePipelines->getLayout(),
        0,
        {
            **cubemapCaptureDescriptorSet,
        },
        nullptr
    );

    for (uint32_t i = 0; i < 6; i++) {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *(*cubemapCapturePipelines)[i]);

        const SkyboxPushConstants pushConstants{
            .view = cubemapFaceViews[i],
        };

        commandBuffer.pushConstants<SkyboxPushConstants>(
            *cubemapCapturePipelines->getLayout(),
            vk::ShaderStageFlagBits::eVertex,
            0u,
            pushConstants
        );

        commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

        if (i != 5) {
            commandBuffer.nextSubpass(vk::SubpassContents::eInline);
        }
    }

    commandBuffer.endRenderPass();

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);

    utils::img::transitionImageLayout(
        ctx,
        **skyboxTexture->getImage(),
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        1,
        6,
        *commandPool,
        *graphicsQueue
    );
}

void VulkanRenderer::captureIrradianceMap() {
    const vk::Extent2D extent{
        .width = irradianceMapTexture->getImage().getExtent().width,
        .height = irradianceMapTexture->getImage().getExtent().height
    };

    constexpr vk::ClearColorValue clearColor{0, 0, 0, 1};
    const std::vector<vk::ClearValue> clearValues{6, clearColor};

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = static_cast<float>(extent.height),
        .width = static_cast<float>(extent.width),
        .height = -1 * static_cast<float>(extent.height), // flip the y-axis
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = extent
    };

    const vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = **envmapConvoluteRenderPass,
        .framebuffer = **irradianceCaptureFramebuffer,
        .renderArea = {
            .offset = {0, 0},
            .extent = extent,
        },
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()
    };

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *irradianceCapturePipelines->getLayout(),
        0,
        {
            **envmapConvoluteDescriptorSet,
        },
        nullptr
    );

    for (size_t i = 0; i < 6; i++) {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *(*irradianceCapturePipelines)[i]);

        const SkyboxPushConstants pushConstants{
            .view = cubemapFaceViews[i],
        };

        commandBuffer.pushConstants<SkyboxPushConstants>(
            *irradianceCapturePipelines->getLayout(),
            vk::ShaderStageFlagBits::eVertex,
            0u,
            pushConstants
        );

        commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

        if (i != 5) {
            commandBuffer.nextSubpass(vk::SubpassContents::eInline);
        }
    }

    commandBuffer.endRenderPass();

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);

    utils::img::transitionImageLayout(
        ctx,
        **irradianceMapTexture->getImage(),
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        1,
        6,
        *commandPool,
        *graphicsQueue
    );
}

void VulkanRenderer::prefilterEnvmap() {
    constexpr vk::ClearColorValue clearColor{0, 0, 0, 1};
    const std::vector<vk::ClearValue> clearValues{6, clearColor};

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    for (uint32_t mipLevel = 0; mipLevel < maxPrefilterMipLevels; mipLevel++) {
        const uint32_t mipScalingFactor = 1 << mipLevel;

        const vk::Extent2D extent{
            .width = prefilteredEnvmapTexture->getImage().getExtent().width / mipScalingFactor,
            .height = prefilteredEnvmapTexture->getImage().getExtent().height / mipScalingFactor
        };

        const vk::Viewport viewport{
            .x = 0.0f,
            .y = static_cast<float>(extent.height),
            .width = static_cast<float>(extent.width),
            .height = -1 * static_cast<float>(extent.height), // flip the y-axis
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = extent
        };

        commandBuffer.setViewport(0, viewport);
        commandBuffer.setScissor(0, scissor);

        const vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = **envmapConvoluteRenderPass,
            .framebuffer = **prefilterFramebuffers[mipLevel],
            .renderArea = {
                .offset = {0, 0},
                .extent = extent,
            },
            .clearValueCount = static_cast<uint32_t>(clearValues.size()),
            .pClearValues = clearValues.data()
        };

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

        commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *prefilterPipelines->getLayout(),
            0,
            {
                **envmapConvoluteDescriptorSet,
            },
            nullptr
        );

        for (size_t i = 0; i < 6; i++) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *(*prefilterPipelines)[i]);

            const PrefilterPushConstants prefilterPushConstants{
                .view = cubemapFaceViews[i],
                .roughness = static_cast<float>(mipLevel) / (maxPrefilterMipLevels - 1)
            };

            commandBuffer.pushConstants<PrefilterPushConstants>(
                *prefilterPipelines->getLayout(),
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                0u,
                prefilterPushConstants
            );

            commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

            if (i != 5) {
                commandBuffer.nextSubpass(vk::SubpassContents::eInline);
            }
        }

        commandBuffer.endRenderPass();
    }

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);

    utils::img::transitionImageLayout(
        ctx,
        **prefilteredEnvmapTexture->getImage(),
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        maxPrefilterMipLevels,
        6,
        *commandPool,
        *graphicsQueue
    );
}

void VulkanRenderer::updateGraphicsUniformBuffer() const {
    const glm::mat4 model = glm::translate(
        glm::scale(
            glm::identity<glm::mat4>(),
            glm::vec3(modelScale)
        ),
        modelTranslate
    );
    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 proj = camera->getProjectionMatrix();

    glm::vec<2, int> windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    const GraphicsUBO graphicsUbo{
        .window = {
            .windowWidth = static_cast<uint32_t>(windowSize.x),
            .windowHeight = static_cast<uint32_t>(windowSize.y),
        },
        .matrices = {
            .model = model,
            .view = view,
            .proj = proj,
            .inverseVp = glm::inverse(proj * view),
            .staticView = camera->getStaticViewMatrix(),
            .cubemapCaptureProj = cubemapFaceProjection
        },
        .misc = {
            .useIBL = useIBL ? 1u : 0u,
            .cameraPos = camera->getPos(),
            .lightDir = glm::normalize(glm::vec3(1, 1.5, -2)),
        }
    };

    memcpy(frameResources[currentFrameIdx].graphicsUboMapped, &graphicsUbo, sizeof(graphicsUbo));
}
