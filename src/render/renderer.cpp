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
#include "vk/descriptor.h"
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

    swapChain = make_unique<SwapChain>(
        ctx,
        *surface,
        findQueueFamilies(*ctx.physicalDevice),
        window,
        msaaSampleCount
    );

    createCommandPool();
    createCommandBuffers();

    createDescriptorPool();

    createUniformBuffers();
    updateGraphicsUniformBuffer();

    createDebugQuadDescriptorSet();
    createDebugQuadPipeline();

    createPrepassTextures();
    createPrepassRenderInfo();
    createPrepassDescriptorSets();
    createPrepassPipeline();

    createIblTextures();

    createSkyboxVertexBuffer();
    createSkyboxDescriptorSets();
    createSkyboxPipeline();

    createCubemapCaptureRenderInfo();
    createCubemapCaptureDescriptorSet();
    createCubemapCapturePipeline();

    createEnvmapConvoluteDescriptorSet();

    createIrradianceCaptureRenderInfo();
    createIrradianceCapturePipeline();

    createPrefilterRenderInfo();
    createPrefilterPipeline();

    createScreenSpaceQuadVertexBuffer();
    createBrdfIntegrationRenderInfo();
    createBrdfIntegrationPipeline();
    computeBrdfIntegrationMap();

    createSceneDescriptorSets();
    createScenePipeline();

    // loadModel("../assets/t-60-helmet/helmet.fbx");
    // loadAlbedoTexture("../assets/t-60-helmet/albedo.png");
    // loadNormalMap("../assets/t-60-helmet/normal.png");
    // //loadOrmMap("../assets/t-60-helmet/orm.png");
    // loadOrmMap(
    //     "",
    //     "../assets/t-60-helmet/roughness.png",
    //     "../assets/t-60-helmet/metallic.png"
    // );

    loadModel("../assets/czajnik/czajnik.obj");
    loadAlbedoTexture("../assets/czajnik/czajnik-albedo.png");
    loadNormalMap("../assets/czajnik/czajnik-normal.png");
    loadOrmMap("../assets/czajnik/czajnik-orm.png");

    loadEnvironmentMap("../assets/envmaps/gallery.hdr");

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

        const auto viewVectors = camera->getViewVectors();

        modelTranslate += cameraDistance * speed * viewVectors.right * static_cast<float>(dx);
        modelTranslate -= cameraDistance * speed * viewVectors.up * static_cast<float>(dy);
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
    if (!supportedFeatures.samplerAnisotropy || !supportedFeatures.fillModeNonSolid) {
        return false;
    }

    const auto supportedFeatures2Chain = physicalDevice.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceSynchronization2FeaturesKHR,
        vk::PhysicalDeviceMultiviewFeatures,
        vk::PhysicalDeviceDynamicRenderingFeatures>();

    const auto vulkan12Features = supportedFeatures2Chain.get<vk::PhysicalDeviceVulkan12Features>();
    if (!vulkan12Features.timelineSemaphore) {
        return false;
    }

    const auto sync2Features = supportedFeatures2Chain.get<vk::PhysicalDeviceSynchronization2FeaturesKHR>();
    if (!sync2Features.synchronization2) {
        return false;
    }

    const auto multiviewFeatures = supportedFeatures2Chain.get<vk::PhysicalDeviceMultiviewFeatures>();
    if (!multiviewFeatures.multiview) {
        return false;
    }

    const auto dynamicRenderFeatures = supportedFeatures2Chain.get<vk::PhysicalDeviceDynamicRenderingFeatures>();
    if (!dynamicRenderFeatures.dynamicRendering) {
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
        .fillModeNonSolid = vk::True,
        .samplerAnisotropy = vk::True,
    };

    vk::PhysicalDeviceMultiviewFeatures multiviewFeatures{
        .multiview = vk::True,
    };

    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderFeatures{
        .pNext = &multiviewFeatures,
        .dynamicRendering = vk::True,
    };

    vk::PhysicalDeviceSynchronization2FeaturesKHR sync2Features{
        .pNext = &dynamicRenderFeatures,
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

    createModelVertexBuffer();
    createIndexBuffer();
}

// ==================== assets ====================

void VulkanRenderer::loadAlbedoTexture(const std::filesystem::path &path) {
    waitIdle();

    albedoTexture.reset();
    albedoTexture = TextureBuilder()
            .fromPaths({path})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->updateBinding(ctx, 1, *albedoTexture);
    }
}

void VulkanRenderer::loadNormalMap(const std::filesystem::path &path) {
    waitIdle();

    normalTexture.reset();
    normalTexture = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx, *commandPool, *graphicsQueue);

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->updateBinding(ctx, 2, *normalTexture);
        res.prepassDescriptorSet->updateBinding(ctx, 1, *normalTexture);
    }
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &path) {
    waitIdle();

    ormTexture.reset();
    ormTexture = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx, *commandPool, *graphicsQueue);

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->updateBinding(ctx, 3, *ormTexture);
    }
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &aoPath, const std::filesystem::path &roughnessPath,
                                const std::filesystem::path &metallicPath) {
    waitIdle();

    ormTexture.reset();
    ormTexture = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .asSeparateChannels()
            .fromPaths({aoPath, roughnessPath, metallicPath})
            .withSwizzle({
                aoPath.empty() ? SwizzleComp::MAX : SwizzleComp::R,
                SwizzleComp::G,
                metallicPath.empty() ? SwizzleComp::ZERO : SwizzleComp::B,
                SwizzleComp::A
            })
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->updateBinding(ctx, 3, *ormTexture);
    }
}

void VulkanRenderer::loadRmaMap(const std::filesystem::path &path) {
    waitIdle();

    ormTexture.reset();
    ormTexture = TextureBuilder()
            .withSwizzle({SwizzleComp::B, SwizzleComp::R, SwizzleComp::G, SwizzleComp::A})
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx, *commandPool, *graphicsQueue);

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->updateBinding(ctx, 3, *ormTexture);
    }
}

void VulkanRenderer::loadEnvironmentMap(const std::filesystem::path &path) {
    waitIdle();

    envmapTexture = TextureBuilder()
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .fromPaths({path})
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    cubemapCaptureDescriptorSet->updateBinding(ctx, 1, *envmapTexture);

    captureCubemap();
    captureIrradianceMap();
    prefilterEnvmap();
}

void VulkanRenderer::createPrepassTextures() {
    const auto &[width, height] = swapChain->getExtent();

    const vk::Extent3D extent{
        .width = width,
        .height = height,
        .depth = 1
    };

    gBufferTextures.normal = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(prepassNormalFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx, *commandPool, *graphicsQueue);

    gBufferTextures.depth = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(swapChain->getDepthFormat())
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .create(ctx, *commandPool, *graphicsQueue);

    if (debugQuadDescriptorSet) {
        debugQuadDescriptorSet->updateBinding(ctx, 0, *gBufferTextures.depth);
    }

    for (auto &res: frameResources) {
        if (res.sceneDescriptorSet) {
            res.sceneDescriptorSet->queueUpdate(7, *gBufferTextures.normal)
                    .queueUpdate(8, *gBufferTextures.depth)
                    .commitUpdates(ctx);
        }
    }
}

void VulkanRenderer::createIblTextures() {
    const auto attachmentUsageFlags = vk::ImageUsageFlagBits::eTransferSrc
                                      | vk::ImageUsageFlagBits::eTransferDst
                                      | vk::ImageUsageFlagBits::eSampled
                                      | vk::ImageUsageFlagBits::eColorAttachment;

    skyboxTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({2048, 2048, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(attachmentUsageFlags)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    irradianceMapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({64, 64, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(attachmentUsageFlags)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    prefilteredEnvmapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({128, 128, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(attachmentUsageFlags)
            .makeMipmaps()
            .create(ctx, *commandPool, *graphicsQueue);

    brdfIntegrationMapTexture = TextureBuilder()
            .asUninitialized({512, 512, 1})
            .useFormat(brdfIntegrationMapFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
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

    waitIdle();

    swapChain.reset();
    swapChain = make_unique<SwapChain>(
        ctx,
        *surface,
        findQueueFamilies(*ctx.physicalDevice),
        window,
        msaaSampleCount
    );

    createPrepassTextures();
    createPrepassRenderInfo();
}

// ==================== descriptors ====================

void VulkanRenderer::createDescriptorPool() {
    static constexpr vk::DescriptorPoolSize uboPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 3 + 2,
    };

    static constexpr vk::DescriptorPoolSize samplerPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 10 + 3,
    };

    static constexpr std::array poolSizes = {
        uboPoolSize,
        samplerPoolSize
    };

    static constexpr vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 3 + 3,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    descriptorPool = make_unique<vk::raii::DescriptorPool>(*ctx.device, poolInfo);
}

void VulkanRenderer::createSceneDescriptorSets() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            )
            .addRepeatedBindings(8, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        frameResources[i].sceneDescriptorSet = make_unique<DescriptorSet>(std::move(sets[i]));
    }

    for (auto &res: frameResources) {
        res.sceneDescriptorSet->queueUpdate(
                    0,
                    *res.graphicsUniformBuffer,
                    vk::DescriptorType::eUniformBuffer,
                    sizeof(GraphicsUBO)
                )
                .queueUpdate(4, *irradianceMapTexture)
                .queueUpdate(5, *prefilteredEnvmapTexture)
                .queueUpdate(6, *brdfIntegrationMapTexture)
                .queueUpdate(7, *gBufferTextures.normal)
                .queueUpdate(8, *gBufferTextures.depth)
                .commitUpdates(ctx);
    }
}

void VulkanRenderer::createSkyboxDescriptorSets() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            )
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        frameResources[i].skyboxDescriptorSet = make_unique<DescriptorSet>(std::move(sets[i]));
    }

    for (auto &res: frameResources) {
        res.skyboxDescriptorSet->queueUpdate(
                    0,
                    *res.graphicsUniformBuffer,
                    vk::DescriptorType::eUniformBuffer,
                    sizeof(GraphicsUBO)
                )
                .queueUpdate(1, *skyboxTexture)
                .commitUpdates(ctx);
    }
}

void VulkanRenderer::createPrepassDescriptorSets() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            )
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        frameResources[i].prepassDescriptorSet = make_unique<DescriptorSet>(std::move(sets[i]));
    }

    for (auto &res: frameResources) {
        res.prepassDescriptorSet->updateBinding(
            ctx,
            0,
            *res.graphicsUniformBuffer,
            vk::DescriptorType::eUniformBuffer,
            sizeof(GraphicsUBO)
        );
    }
}

void VulkanRenderer::createCubemapCaptureDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    cubemapCaptureDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    cubemapCaptureDescriptorSet->updateBinding(
        ctx,
        0,
        *frameResources[0].graphicsUniformBuffer,
        vk::DescriptorType::eUniformBuffer,
        sizeof(GraphicsUBO)
    );
}

void VulkanRenderer::createEnvmapConvoluteDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    envmapConvoluteDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    envmapConvoluteDescriptorSet->queueUpdate(
                0,
                *frameResources[0].graphicsUniformBuffer,
                vk::DescriptorType::eUniformBuffer,
                sizeof(GraphicsUBO)
            )
            .queueUpdate(1, *skyboxTexture)
            .commitUpdates(ctx);
}

void VulkanRenderer::createDebugQuadDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = utils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    debugQuadDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    if (gBufferTextures.depth) {
        debugQuadDescriptorSet->updateBinding(ctx, 0, *gBufferTextures.depth);
    }
}

// ==================== render infos ====================

vk::RenderingInfo RenderInfo::get(const vk::Extent2D extent, const uint32_t views,
                                  const vk::RenderingFlags flags) const {
    return {
        .flags = flags,
        .renderArea = {
            .offset = {0, 0},
            .extent = extent
        },
        .layerCount = views == 1 ? 1u : 0u,
        .viewMask = views == 1 ? 0 : (1u << views) - 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
        .pDepthAttachment = depthAttachment ? &depthAttachment.value() : nullptr
    };
}

void VulkanRenderer::createPrepassRenderInfo() {
    const std::vector colorAttachments{
        vk::RenderingAttachmentInfo{
            .imageView = *gBufferTextures.normal->getView(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        }
    };

    const vk::RenderingAttachmentInfo depthAttachment{
        .imageView = *gBufferTextures.depth->getView(),
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearDepthStencilValue{
            .depth = 1.0f,
            .stencil = 0,
        },
    };

    prepassRenderInfo = {
        .colorAttachments = colorAttachments,
        .depthAttachment = depthAttachment
    };
}

void VulkanRenderer::createCubemapCaptureRenderInfo() {
    const std::vector colorAttachments{
        vk::RenderingAttachmentInfo{
            .imageView = *skyboxTexture->getAttachmentView(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        }
    };

    cubemapCaptureRenderInfo = {
        .colorAttachments = colorAttachments,
    };
}

void VulkanRenderer::createIrradianceCaptureRenderInfo() {
    const std::vector colorAttachments{
        vk::RenderingAttachmentInfo{
            .imageView = *irradianceMapTexture->getAttachmentView(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        }
    };

    irradianceCaptureRenderInfo = {
        .colorAttachments = colorAttachments,
    };
}

void VulkanRenderer::createPrefilterRenderInfo() {
    for (uint32_t i = 0; i < maxPrefilterMipLevels; i++) {
        const std::vector colorAttachments{
            vk::RenderingAttachmentInfo{
                .imageView = *prefilteredEnvmapTexture->getMipView(i),
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
            }
        };

        prefilterRenderInfos.emplace_back(RenderInfo{
            .colorAttachments = colorAttachments,
        });
    }
}

void VulkanRenderer::createBrdfIntegrationRenderInfo() {
    const std::vector colorAttachments{
        vk::RenderingAttachmentInfo{
            .imageView = *brdfIntegrationMapTexture->getAttachmentView(),
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        }
    };

    brdfIntegrationRenderInfo = {
        .colorAttachments = colorAttachments,
    };
}

// ==================== pipelines ====================

void VulkanRenderer::createScenePipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/main-vert.spv")
            .withFragmentShader("../shaders/obj/main-frag.spv")
            .withVertices<Vertex>()
            .withRasterizer({
                .polygonMode = wireframeMode ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                .cullMode = cullBackFaces ? vk::CullModeFlagBits::eBack : vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withMultisampling({
                .rasterizationSamples = msaaSampleCount,
                .minSampleShading = 1.0f,
            })
            .withDescriptorLayouts({
                *frameResources[0].sceneDescriptorSet->getLayout(),
            })
            .withColorFormats({swapChain->getImageFormat()})
            .withDepthFormat(swapChain->getDepthFormat())
            .create(ctx);

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
                *frameResources[0].skyboxDescriptorSet->getLayout(),
            })
            .withColorFormats({swapChain->getImageFormat()})
            .withDepthFormat(swapChain->getDepthFormat())
            .create(ctx);

    skyboxPipeline = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createPrepassPipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/prepass-vert.spv")
            .withFragmentShader("../shaders/obj/prepass-frag.spv")
            .withVertices<Vertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDescriptorLayouts({
                *frameResources[0].prepassDescriptorSet->getLayout(),
            })
            .withColorFormats({gBufferTextures.normal->getFormat()})
            .withDepthFormat(swapChain->getDepthFormat())
            .create(ctx);

    prepassPipeline = make_unique<PipelinePack>(std::move(pipeline));
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
                *cubemapCaptureDescriptorSet->getLayout(),
            })
            .forViews(6)
            .withColorFormats({hdrEnvmapFormat})
            .create(ctx);

    cubemapCapturePipeline = make_unique<PipelinePack>(std::move(pipeline));
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
                *envmapConvoluteDescriptorSet->getLayout(),
            })
            .forViews(6)
            .withColorFormats({hdrEnvmapFormat})
            .create(ctx);

    irradianceCapturePipeline = make_unique<PipelinePack>(std::move(pipeline));
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
                *envmapConvoluteDescriptorSet->getLayout(),
            })
            .withPushConstants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                    .offset = 0,
                    .size = sizeof(PrefilterPushConstants),
                }
            })
            .forViews(6)
            .withColorFormats({hdrEnvmapFormat})
            .create(ctx);

    prefilterPipeline = make_unique<PipelinePack>(std::move(pipeline));
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
            .withColorFormats({brdfIntegrationMapFormat})
            .create(ctx);

    brdfIntegrationPipeline = make_unique<PipelinePack>(std::move(pipeline));
}

void VulkanRenderer::createDebugQuadPipeline() {
    PipelinePack pipeline = PipelineBuilder()
            .withVertexShader("../shaders/obj/ss-quad-vert.spv")
            .withFragmentShader("../shaders/obj/ss-quad-frag.spv")
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
            .withDescriptorLayouts({
                *debugQuadDescriptorSet->getLayout(),
            })
            .withColorFormats({swapChain->getImageFormat()})
            .withDepthFormat(swapChain->getDepthFormat())
            .create(ctx);

    debugQuadPipeline = make_unique<PipelinePack>(std::move(pipeline));
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

void VulkanRenderer::createModelVertexBuffer() {
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
    vk::raii::CommandBuffers prepassCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers debugCommandBuffers{*ctx.device, secondaryAllocInfo};

    for (size_t i = 0; i < graphicsCommandBuffers.size(); i++) {
        frameResources[i].graphicsCmdBuffer =
                make_unique<vk::raii::CommandBuffer>(std::move(graphicsCommandBuffers[i]));
        frameResources[i].sceneCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(sceneCommandBuffers[i]))};
        frameResources[i].guiCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(guiCommandBuffers[i]))};
        frameResources[i].prepassCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(prepassCommandBuffers[i]))};
        frameResources[i].debugCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(debugCommandBuffers[i]))};
    }
}

void VulkanRenderer::recordGraphicsCommandBuffer() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].graphicsCmdBuffer;

    constexpr auto renderingFlags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;

    constexpr vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);

    swapChain->transitionToAttachmentLayout(commandBuffer);

    // prepass

    if (frameResources[currentFrameIdx].prepassCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.beginRendering(prepassRenderInfo.get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].prepassCmdBuffer);
        commandBuffer.endRendering();
    }

    // main pass

    if (frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.beginRendering(swapChain->getRenderInfo().get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].sceneCmdBuffer);
        commandBuffer.endRendering();
    }

    // debug quad pass

    if (frameResources[currentFrameIdx].debugCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.beginRendering(swapChain->getRenderInfo().get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].debugCmdBuffer);
        commandBuffer.endRendering();
    }

    // gui pass

    if (frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.beginRendering(swapChain->getGuiRenderInfo().get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].guiCmdBuffer);
        commandBuffer.endRendering();
    }

    swapChain->transitionToPresentLayout(commandBuffer);

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
        .UseDynamicRendering = true,
        .ColorAttachmentFormat = static_cast<VkFormat>(swapChain->getImageFormat()),
    };

    guiRenderer = make_unique<GuiRenderer>(window, imguiInitInfo);
}

void VulkanRenderer::renderGuiSection() {
    constexpr auto sectionFlags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Model ", sectionFlags)) {
        if (ImGui::Button("Load model...")) {
            ImGui::OpenPopup("Load model");
        }

        ImGui::Separator();

        ImGui::DragFloat("Model scale", &modelScale, 0.01, 0, std::numeric_limits<float>::max());

        ImGui::gizmo3D("Model rotation", modelRotation, 160);

        if (ImGui::Button("Reset scale")) { modelScale = 1; }
        ImGui::SameLine();
        if (ImGui::Button("Reset rotation")) { modelRotation = {1, 0, 0, 0}; }
        ImGui::SameLine();
        if (ImGui::Button("Reset position")) { modelTranslate = {0, 0, 0}; }

        ImGui::Separator();

        if (ImGui::Checkbox("Cull backfaces", &cullBackFaces)) {
            waitIdle();
            createScenePipeline();
        }

        if (ImGui::Checkbox("Wireframe mode", &wireframeMode)) {
            waitIdle();
            createScenePipeline();
        }

        // ImGui::DragFloat("debug number", &debugNumber, 0.01, 0, std::numeric_limits<float>::max());
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

    const std::vector colorAttachmentFormats{swapChain->getImageFormat()};

    const vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
        .pColorAttachmentFormats = colorAttachmentFormats.data(),
        .rasterizationSamples = msaaSampleCount,
    };

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .pNext = &inheritanceRenderingInfo
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    guiRenderer->beginRendering();
    renderCommands();
    guiRenderer->endRendering(commandBuffer);

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
        throw std::runtime_error("waitSemaphores on renderFinishedTimeline failed");
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

    frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame = false;
    frameResources[currentFrameIdx].prepassCmdBuffer.wasRecordedThisFrame = false;
    frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame = false;
    frameResources[currentFrameIdx].debugCmdBuffer.wasRecordedThisFrame = false;

    return true;
}

void VulkanRenderer::endFrame() {
    recordGraphicsCommandBuffer();

    auto &sync = frameResources[currentFrameIdx].sync;

    const std::vector waitSemaphores = {
        **sync.imageAvailableSemaphore
    };

    const std::vector<TimelineSemValueType> waitSemaphoreValues = {
        0
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
    const std::vector<TimelineSemValueType> signalSemaphoreValues{
        sync.renderFinishedTimeline.timeline,
        0
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
        .commandBufferCount = 1,
        .pCommandBuffers = &**frameResources[currentFrameIdx].graphicsCmdBuffer,
        .signalSemaphoreCount = signalSemaphores.size(),
        .pSignalSemaphores = signalSemaphores.data(),
    };

    try {
        graphicsQueue->submit(graphicsSubmitInfo);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        throw e;
    }

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

void VulkanRenderer::runPrepass() {
    if (!model) {
        return;
    }

    const auto &commandBuffer = *frameResources[currentFrameIdx].prepassCmdBuffer.buffer;

    const std::vector colorAttachmentFormats{gBufferTextures.normal->getFormat()};

    const vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
        .pColorAttachmentFormats = colorAttachmentFormats.data(),
        .depthAttachmentFormat = swapChain->getDepthFormat(),
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .pNext = &inheritanceRenderingInfo
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    utils::cmd::setDynamicStates(commandBuffer, swapChain->getExtent());

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***prepassPipeline);

    commandBuffer.bindVertexBuffers(0, **vertexBuffer, {0});
    commandBuffer.bindVertexBuffers(1, **instanceDataBuffer, {0});
    commandBuffer.bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *prepassPipeline->getLayout(),
        0,
        ***frameResources[currentFrameIdx].prepassDescriptorSet,
        nullptr
    );

    drawModel(commandBuffer);

    commandBuffer.end();

    frameResources[currentFrameIdx].prepassCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawScene() {
    if (!model) {
        return;
    }

    const auto &commandBuffer = *frameResources[currentFrameIdx].sceneCmdBuffer.buffer;

    const vk::Extent2D swapChainExtent = swapChain->getExtent();

    const std::vector colorAttachmentFormats{swapChain->getImageFormat()};

    const vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
        .pColorAttachmentFormats = colorAttachmentFormats.data(),
        .depthAttachmentFormat = swapChain->getDepthFormat(),
        .rasterizationSamples = msaaSampleCount,
    };

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .pNext = &inheritanceRenderingInfo
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    utils::cmd::setDynamicStates(commandBuffer, swapChainExtent);

    // skybox

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***skyboxPipeline);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *skyboxPipeline->getLayout(),
        0,
        ***frameResources[currentFrameIdx].skyboxDescriptorSet,
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
        ***frameResources[currentFrameIdx].sceneDescriptorSet,
        nullptr
    );

    drawModel(commandBuffer);

    commandBuffer.end();

    frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawDebugQuad() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].debugCmdBuffer.buffer;

    const vk::Extent2D swapChainExtent = swapChain->getExtent();

    const std::vector colorAttachmentFormats{swapChain->getImageFormat()};

    const vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
        .pColorAttachmentFormats = colorAttachmentFormats.data(),
        .depthAttachmentFormat = swapChain->getDepthFormat(),
        .rasterizationSamples = msaaSampleCount,
    };

    const vk::CommandBufferInheritanceInfo inheritanceInfo{
        .pNext = &inheritanceRenderingInfo
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo,
    };

    commandBuffer.begin(beginInfo);

    utils::cmd::setDynamicStates(commandBuffer, swapChainExtent);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***debugQuadPipeline);

    commandBuffer.bindVertexBuffers(0, **screenSpaceQuadVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *debugQuadPipeline->getLayout(),
        0,
        ***debugQuadDescriptorSet,
        nullptr
    );

    commandBuffer.draw(screenSpaceQuadVertices.size(), 1, 0, 0);

    commandBuffer.end();

    frameResources[currentFrameIdx].debugCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawModel(const vk::raii::CommandBuffer &commandBuffer) const {
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

void VulkanRenderer::captureCubemap() const {
    const vk::Extent2D extent = skyboxTexture->getImage().getExtent2d();

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    utils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(cubemapCaptureRenderInfo.get(extent, 6));

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *cubemapCapturePipeline->getLayout(),
        0,
        ***cubemapCaptureDescriptorSet,
        nullptr
    );

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***cubemapCapturePipeline);

    commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    skyboxTexture->getImage().transitionLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        commandBuffer
    );

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);

    skyboxTexture->generateMipmaps(
        ctx,
        *commandPool,
        *graphicsQueue,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );
}

void VulkanRenderer::captureIrradianceMap() const {
    const vk::Extent2D extent = irradianceMapTexture->getImage().getExtent2d();

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    utils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(irradianceCaptureRenderInfo.get(extent, 6));

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *irradianceCapturePipeline->getLayout(),
        0,
        ***envmapConvoluteDescriptorSet,
        nullptr
    );

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***irradianceCapturePipeline);

    commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    irradianceMapTexture->getImage().transitionLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        commandBuffer
    );

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);

    irradianceMapTexture->generateMipmaps(
        ctx,
        *commandPool,
        *graphicsQueue,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );
}

void VulkanRenderer::prefilterEnvmap() const {
    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    for (uint32_t mipLevel = 0; mipLevel < maxPrefilterMipLevels; mipLevel++) {
        const uint32_t mipScalingFactor = 1 << mipLevel;

        vk::Extent2D extent = prefilteredEnvmapTexture->getImage().getExtent2d();
        extent.width /= mipScalingFactor;
        extent.height /= mipScalingFactor;

        utils::cmd::setDynamicStates(commandBuffer, extent);

        commandBuffer.beginRendering(prefilterRenderInfos[mipLevel].get(extent, 6));

        commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *prefilterPipeline->getLayout(),
            0,
            ***envmapConvoluteDescriptorSet,
            nullptr
        );

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***prefilterPipeline);

        const PrefilterPushConstants prefilterPushConstants{
            .roughness = static_cast<float>(mipLevel) / (maxPrefilterMipLevels - 1)
        };

        commandBuffer.pushConstants<PrefilterPushConstants>(
            *prefilterPipeline->getLayout(),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0u,
            prefilterPushConstants
        );

        commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

        commandBuffer.endRendering();
    }

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);
}

void VulkanRenderer::computeBrdfIntegrationMap() const {
    const vk::Extent2D extent = brdfIntegrationMapTexture->getImage().getExtent2d();

    const auto commandBuffer = utils::cmd::beginSingleTimeCommands(*ctx.device, *commandPool);

    utils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(brdfIntegrationRenderInfo.get(extent));

    commandBuffer.bindVertexBuffers(0, **screenSpaceQuadVertexBuffer, {0});

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ***brdfIntegrationPipeline);

    commandBuffer.draw(screenSpaceQuadVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    utils::cmd::endSingleTimeCommands(commandBuffer, *graphicsQueue);
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

void VulkanRenderer::updateGraphicsUniformBuffer() const {
    const glm::mat4 model = glm::translate(modelTranslate)
                            * mat4_cast(modelRotation)
                            * glm::scale(glm::vec3(modelScale));
    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 proj = camera->getProjectionMatrix();

    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    GraphicsUBO graphicsUbo{
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
            .debugNumber = debugNumber,
            .cameraPos = camera->getPos(),
            .lightDir = glm::normalize(glm::vec3(1, 1.5, -2)),
        }
    };

    for (size_t i = 0; i < 6; i++) {
        graphicsUbo.matrices.cubemapCaptureViews[i] = cubemapFaceViews[i];
    }

    memcpy(frameResources[currentFrameIdx].graphicsUboMapped, &graphicsUbo, sizeof(graphicsUbo));
}
