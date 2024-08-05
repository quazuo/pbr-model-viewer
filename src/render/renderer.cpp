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
#include <random>

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
        getMsaaSampleCount()
    );

    createCommandPool();
    createCommandBuffers();

    createDescriptorPool();

    createUniformBuffers();
    updateGraphicsUniformBuffer();

    createDebugQuadDescriptorSet();
    createDebugQuadRenderInfos();

    createPrepassTextures();
    createPrepassDescriptorSets();
    createPrepassRenderInfo();

    createSsaoTextures();
    createSsaoDescriptorSets();
    createSsaoRenderInfo();

    createIblTextures();
    createIblDescriptorSet();

    createSkyboxVertexBuffer();
    createSkyboxDescriptorSets();
    createSkyboxRenderInfos();

    createCubemapCaptureDescriptorSet();
    createCubemapCaptureRenderInfo();

    createEnvmapConvoluteDescriptorSet();
    createIrradianceCaptureRenderInfo();
    createPrefilterRenderInfos();

    createScreenSpaceQuadVertexBuffer();
    createBrdfIntegrationRenderInfo();
    computeBrdfIntegrationMap();

    createMaterialsDescriptorSet();
    createSceneDescriptorSets();
    createSceneRenderInfos();
    createGuiRenderInfos();

    loadModelWithMaterials("../assets/example models/sponza/Sponza.gltf");

    // loadModel("../assets/example models/kettle/kettle.obj");
    // loadBaseColorTexture("../assets/example models/kettle/kettle-albedo.png");
    // loadNormalMap("../assets/example models/kettle/kettle-normal.png");
    // loadOrmMap("../assets/example models/kettle/kettle-orm.png");

    loadEnvironmentMap("../assets/envmaps/vienna.hdr");

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

    if (enableValidationLayers) {
        const vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfo{
            {
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
                .ppEnabledLayerNames = validationLayers.data(),
                .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
                .ppEnabledExtensionNames = extensions.data(),
            },
            makeDebugMessengerCreateInfo()
        };

        instance = make_unique<vk::raii::Instance>(vkCtx, instanceCreateInfo.get<vk::InstanceCreateInfo>());

    } else {
        const vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        instance = make_unique<vk::raii::Instance>(vkCtx, createInfo);
    }
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

    const vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceSynchronization2FeaturesKHR,
        vk::PhysicalDeviceDynamicRenderingFeatures,
        vk::PhysicalDeviceMultiviewFeatures
    > createInfo{
        {
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledLayerCount = static_cast<uint32_t>(enableValidationLayers ? validationLayers.size() : 0),
            .ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures,
        },
        {
            .timelineSemaphore = vk::True,
        },
        {
            .synchronization2 = vk::True,
        },
        {
            .dynamicRendering = vk::True,
        },
        {
            .multiview = vk::True,
        }
    };

    ctx.device = make_unique<vk::raii::Device>(*ctx.physicalDevice, createInfo.get<vk::DeviceCreateInfo>());

    ctx.graphicsQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(graphicsComputeFamily.value(), 0));
    presentQueue = make_unique<vk::raii::Queue>(ctx.device->getQueue(presentFamily.value(), 0));
}

// ==================== models ====================

void VulkanRenderer::loadModelWithMaterials(const std::filesystem::path &path) {
    waitIdle();

    model.reset();
    model = make_unique<Model>(ctx, path, true);

    vertexBuffer.reset();
    indexBuffer.reset();

    createModelVertexBuffer();
    createIndexBuffer();

    const auto &materials = model->getMaterials();

    for (uint32_t i = 0; i < materials.size(); i++) {
        const auto &material = materials[i];

        if (material.baseColor) {
            materialsDescriptorSet->queueUpdate(ctx, 0, *material.baseColor, i);
        }

        if (material.normal) {
            materialsDescriptorSet->queueUpdate(ctx, 1, *material.normal, i);
        }

        if (material.orm) {
            materialsDescriptorSet->queueUpdate(ctx, 2, *material.orm, i);
        }
    }

    materialsDescriptorSet->commitUpdates(ctx);
}

void VulkanRenderer::loadModel(const std::filesystem::path &path) {
    waitIdle();

    model.reset();
    model = make_unique<Model>(ctx, path, false);

    vertexBuffer.reset();
    indexBuffer.reset();

    createModelVertexBuffer();
    createIndexBuffer();
}

// ==================== assets ====================

void VulkanRenderer::loadBaseColorTexture(const std::filesystem::path &path) {
    waitIdle();

    separateMaterial.baseColor.reset();
    separateMaterial.baseColor = TextureBuilder()
            .fromPaths({path})
            .makeMipmaps()
            .create(ctx);

    materialsDescriptorSet->updateBinding(ctx, 0, *separateMaterial.baseColor);
}

void VulkanRenderer::loadNormalMap(const std::filesystem::path &path) {
    waitIdle();

    separateMaterial.normal.reset();
    separateMaterial.normal = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx);

    materialsDescriptorSet->updateBinding(ctx, 1, *separateMaterial.normal);

    for (auto &res: frameResources) {
        res.prepassDescriptorSet->updateBinding(ctx, 1, *separateMaterial.normal);
    }
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &path) {
    waitIdle();

    separateMaterial.orm.reset();
    separateMaterial.orm = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx);

    materialsDescriptorSet->updateBinding(ctx, 2, *separateMaterial.orm);
}

void VulkanRenderer::loadOrmMap(const std::filesystem::path &aoPath, const std::filesystem::path &roughnessPath,
                                const std::filesystem::path &metallicPath) {
    waitIdle();

    separateMaterial.orm.reset();
    separateMaterial.orm = TextureBuilder()
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .asSeparateChannels()
            .fromPaths({aoPath, roughnessPath, metallicPath})
            .withSwizzle({
                aoPath.empty() ? SwizzleComponent::MAX : SwizzleComponent::R,
                SwizzleComponent::G,
                metallicPath.empty() ? SwizzleComponent::ZERO : SwizzleComponent::B,
                SwizzleComponent::A
            })
            .makeMipmaps()
            .create(ctx);

    materialsDescriptorSet->updateBinding(ctx, 2, *separateMaterial.orm);
}

void VulkanRenderer::loadRmaMap(const std::filesystem::path &path) {
    waitIdle();

    separateMaterial.orm.reset();
    separateMaterial.orm = TextureBuilder()
            .withSwizzle({SwizzleComponent::B, SwizzleComponent::R, SwizzleComponent::G, SwizzleComponent::A})
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .fromPaths({path})
            .create(ctx);

    materialsDescriptorSet->updateBinding(ctx, 2, *separateMaterial.orm);
}

void VulkanRenderer::loadEnvironmentMap(const std::filesystem::path &path) {
    waitIdle();

    envmapTexture = TextureBuilder()
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .fromPaths({path})
            .withSamplerAddressMode(vk::SamplerAddressMode::eClampToEdge)
            .makeMipmaps()
            .create(ctx);

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

    gBufferTextures.pos = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(prepassColorFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);

    gBufferTextures.normal = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(prepassColorFormat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);

    gBufferTextures.depth = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(swapChain->getDepthFormat())
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .create(ctx);

    for (auto &res: frameResources) {
        if (res.ssaoDescriptorSet) {
            res.ssaoDescriptorSet->queueUpdate(ctx, 1, *gBufferTextures.depth)
                    .queueUpdate(ctx, 2, *gBufferTextures.normal)
                    .queueUpdate(ctx, 3, *gBufferTextures.pos)
                    .commitUpdates(ctx);
        }
    }
}

static std::vector<glm::vec4> makeSsaoNoise() {
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
    std::default_random_engine generator;

    std::vector<glm::vec4> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        glm::vec4 noise(
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator) * 2.0 - 1.0,
            0.0f,
            0.0f
        );
        ssaoNoise.push_back(noise);
    }

    return ssaoNoise;
}

void VulkanRenderer::createSsaoTextures() {
    const auto &[width, height] = swapChain->getExtent();

    const vk::Extent3D extent{
        .width = width,
        .height = height,
        .depth = 1
    };

    ssaoTexture = TextureBuilder()
            .asUninitialized(extent)
            .useFormat(vk::Format::eR8G8B8A8Unorm)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .create(ctx);

    auto noise = makeSsaoNoise();

    ssaoNoiseTexture = TextureBuilder()
            .fromMemory(noise.data(), {4, 4, 1})
            .useFormat(vk::Format::eR32G32B32A32Sfloat)
            .useUsage(vk::ImageUsageFlagBits::eTransferSrc
                      | vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled
                      | vk::ImageUsageFlagBits::eColorAttachment)
            .withSamplerAddressMode(vk::SamplerAddressMode::eRepeat)
            .create(ctx);

    if (debugQuadDescriptorSet) {
        debugQuadDescriptorSet->updateBinding(ctx, 0, *ssaoTexture);
    }

    for (auto &res: frameResources) {
        if (res.sceneDescriptorSet) {
            res.sceneDescriptorSet->updateBinding(ctx, 1, *ssaoTexture);
        }

        if (res.ssaoDescriptorSet) {
            res.ssaoDescriptorSet->updateBinding(ctx, 4, *ssaoNoiseTexture);
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
            .create(ctx);

    irradianceMapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({64, 64, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(attachmentUsageFlags)
            .makeMipmaps()
            .create(ctx);

    prefilteredEnvmapTexture = TextureBuilder()
            .asCubemap()
            .asUninitialized({128, 128, 1})
            .asHdr()
            .useFormat(hdrEnvmapFormat)
            .useUsage(attachmentUsageFlags)
            .makeMipmaps()
            .create(ctx);

    brdfIntegrationMapTexture = TextureBuilder()
            .asUninitialized({512, 512, 1})
            .useFormat(brdfIntegrationMapFormat)
            .withSamplerAddressMode(vk::SamplerAddressMode::eClampToEdge)
            .useUsage(attachmentUsageFlags)
            .create(ctx);
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
        getMsaaSampleCount()
    );

    // todo - this shouldn't recreate pipelines
    createSceneRenderInfos();
    createSkyboxRenderInfos();
    createGuiRenderInfos();
    createDebugQuadRenderInfos();

    createPrepassTextures();
    createPrepassRenderInfo();
    createSsaoTextures();
    createSsaoRenderInfo();
}

// ==================== descriptors ====================

void VulkanRenderer::createDescriptorPool() {
    static constexpr vk::DescriptorPoolSize uboPoolSize{
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 100u,
    };

    static constexpr vk::DescriptorPoolSize samplerPoolSize{
        .type = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1000u,
    };

    static constexpr std::array poolSizes = {
        uboPoolSize,
        samplerPoolSize
    };

    static constexpr vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 4 + 5,
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
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment) // ssao
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

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
                .queueUpdate(ctx, 1, *ssaoTexture)
                .commitUpdates(ctx);
    }
}

void VulkanRenderer::createMaterialsDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addRepeatedBindings(
                3,
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                MATERIAL_TEX_ARRAY_SIZE
            )
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    materialsDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));
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
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

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
                .queueUpdate(ctx, 1, *skyboxTexture)
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
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

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

void VulkanRenderer::createSsaoDescriptorSets() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
            )
            .addRepeatedBindings(4, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        frameResources[i].ssaoDescriptorSet = make_unique<DescriptorSet>(std::move(sets[i]));
    }

    for (auto &res: frameResources) {
        res.ssaoDescriptorSet->queueUpdate(
                    0,
                    *res.graphicsUniformBuffer,
                    vk::DescriptorType::eUniformBuffer,
                    sizeof(GraphicsUBO)
                )
                .queueUpdate(ctx, 1, *gBufferTextures.depth)
                .queueUpdate(ctx, 2, *gBufferTextures.normal)
                .queueUpdate(ctx, 3, *gBufferTextures.pos)
                .queueUpdate(ctx, 4, *ssaoNoiseTexture)
                .commitUpdates(ctx);
    }
}

void VulkanRenderer::createIblDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addRepeatedBindings(3, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    iblDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    iblDescriptorSet->queueUpdate(ctx, 0, *irradianceMapTexture)
            .queueUpdate(ctx, 1, *prefilteredEnvmapTexture)
            .queueUpdate(ctx, 2, *brdfIntegrationMapTexture)
            .commitUpdates(ctx);
}

void VulkanRenderer::createCubemapCaptureDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

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
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    envmapConvoluteDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    envmapConvoluteDescriptorSet->queueUpdate(
                0,
                *frameResources[0].graphicsUniformBuffer,
                vk::DescriptorType::eUniformBuffer,
                sizeof(GraphicsUBO)
            )
            .queueUpdate(ctx, 1, *skyboxTexture)
            .commitUpdates(ctx);
}

void VulkanRenderer::createDebugQuadDescriptorSet() {
    auto layout = DescriptorLayoutBuilder()
            .addBinding(vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
            .create(ctx);

    const auto layoutPtr = make_shared<vk::raii::DescriptorSetLayout>(std::move(layout));
    auto sets = vkutils::desc::createDescriptorSets(ctx, *descriptorPool, layoutPtr, 1);

    debugQuadDescriptorSet = make_unique<DescriptorSet>(std::move(sets[0]));

    if (ssaoTexture) {
        debugQuadDescriptorSet->updateBinding(ctx, 0, *ssaoTexture);
    }
}

// ==================== render infos ====================

RenderInfo::RenderInfo(PipelineBuilder builder, shared_ptr<Pipeline> pipeline, std::vector<RenderTarget> colors)
    : cachedPipelineBuilder(std::move(builder)), pipeline(std::move(pipeline)), colorTargets(std::move(colors)) {
    makeAttachmentInfos();
}

RenderInfo::RenderInfo(PipelineBuilder builder, shared_ptr<Pipeline> pipeline,
                       std::vector<RenderTarget> colors, RenderTarget depth)
    : cachedPipelineBuilder(std::move(builder)), pipeline(std::move(pipeline)),
      colorTargets(std::move(colors)), depthTarget(std::move(depth)) {
    makeAttachmentInfos();
}

RenderInfo::RenderInfo(std::vector<RenderTarget> colors) : colorTargets(std::move(colors)) {
    makeAttachmentInfos();
}

RenderInfo::RenderInfo(std::vector<RenderTarget> colors, RenderTarget depth)
    : colorTargets(std::move(colors)), depthTarget(std::move(depth)) {
    makeAttachmentInfos();
}

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

vk::CommandBufferInheritanceRenderingInfo RenderInfo::getInheritanceRenderingInfo() {
    return {
        .colorAttachmentCount = static_cast<uint32_t>(cachedColorAttachmentFormats.size()),
        .pColorAttachmentFormats = cachedColorAttachmentFormats.data(),
        .depthAttachmentFormat = depthTarget->getFormat(),
        .rasterizationSamples = pipeline->getSampleCount(),
    };
}

void RenderInfo::reloadShaders(const RendererContext &ctx) const {
    *pipeline = cachedPipelineBuilder.create(ctx);
}

void RenderInfo::makeAttachmentInfos() {
    for (const auto &target: colorTargets) {
        colorAttachments.emplace_back(target.getAttachmentInfo());
        cachedColorAttachmentFormats.push_back(target.getFormat());
    }

    if (depthTarget) {
        depthAttachment = depthTarget->getAttachmentInfo();
    }
}

void VulkanRenderer::createSceneRenderInfos() {
    sceneRenderInfos.clear();

    auto builder = PipelineBuilder()
            .withVertexShader("../shaders/obj/main-vert.spv")
            .withFragmentShader("../shaders/obj/main-frag.spv")
            .withVertices<ModelVertex>()
            .withRasterizer({
                .polygonMode = wireframeMode ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
                .cullMode = cullBackFaces ? vk::CullModeFlagBits::eBack : vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withMultisampling({
                .rasterizationSamples = getMsaaSampleCount(),
                .minSampleShading = 1.0f,
            })
            .withDescriptorLayouts({
                *frameResources[0].sceneDescriptorSet->getLayout(),
                *materialsDescriptorSet->getLayout(),
                *iblDescriptorSet->getLayout(),
            })
            .withPushConstants({
                vk::PushConstantRange{
                    .stageFlags = vk::ShaderStageFlagBits::eFragment,
                    .offset = 0,
                    .size = sizeof(ScenePushConstants),
                }
            })
            .withColorFormats({swapChain->getImageFormat()})
            .withDepthFormat(swapChain->getDepthFormat());

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    for (auto &target: swapChain->getRenderTargets(ctx)) {
        std::vector<RenderTarget> colorTargets;
        colorTargets.emplace_back(std::move(target.colorTarget));

        target.depthTarget.overrideAttachmentConfig(vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare);

        sceneRenderInfos.emplace_back(
            builder,
            pipeline,
            std::move(colorTargets),
            std::move(target.depthTarget)
        );
    }
}

void VulkanRenderer::createSkyboxRenderInfos() {
    skyboxRenderInfos.clear();

    auto builder = PipelineBuilder()
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
                .rasterizationSamples = getMsaaSampleCount(),
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
            .withDepthFormat(swapChain->getDepthFormat());

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    for (auto &target: swapChain->getRenderTargets(ctx)) {
        std::vector<RenderTarget> colorTargets;
        colorTargets.emplace_back(std::move(target.colorTarget));

        skyboxRenderInfos.emplace_back(
            builder,
            pipeline,
            std::move(colorTargets),
            std::move(target.depthTarget)
        );
    }
}

void VulkanRenderer::createGuiRenderInfos() {
    guiRenderInfos.clear();

    for (auto &target: swapChain->getRenderTargets(ctx)) {
        target.colorTarget.overrideAttachmentConfig(vk::AttachmentLoadOp::eLoad);

        std::vector<RenderTarget> colorTargets;
        colorTargets.emplace_back(std::move(target.colorTarget));

        guiRenderInfos.emplace_back(std::move(colorTargets));
    }
}

void VulkanRenderer::createPrepassRenderInfo() {
    std::vector<RenderTarget> colorTargets;
    colorTargets.emplace_back(ctx, *gBufferTextures.normal);
    colorTargets.emplace_back(ctx, *gBufferTextures.pos);

    RenderTarget depthTarget{ctx, *gBufferTextures.depth};

    std::vector<vk::Format> colorFormats;
    for (const auto &target: colorTargets) colorFormats.emplace_back(target.getFormat());

    auto builder = PipelineBuilder()
            .withVertexShader("../shaders/obj/prepass-vert.spv")
            .withFragmentShader("../shaders/obj/prepass-frag.spv")
            .withVertices<ModelVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDescriptorLayouts({
                *frameResources[0].prepassDescriptorSet->getLayout(),
            })
            .withColorFormats(colorFormats)
            .withDepthFormat(depthTarget.getFormat());

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    prepassRenderInfo = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(colorTargets),
        std::move(depthTarget)
    );
}

void VulkanRenderer::createSsaoRenderInfo() {
    RenderTarget target{ctx, *ssaoTexture};

    auto builder = PipelineBuilder()
            .withVertexShader("../shaders/obj/ssao-vert.spv")
            .withFragmentShader("../shaders/obj/ssao-frag.spv")
            .withVertices<ScreenSpaceQuadVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withDescriptorLayouts({
                *frameResources[0].ssaoDescriptorSet->getLayout(),
            })
            .withColorFormats({target.getFormat()});

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    ssaoRenderInfo = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::createCubemapCaptureRenderInfo() {
    RenderTarget target{
        skyboxTexture->getImage().getMipView(ctx, 0),
        skyboxTexture->getFormat()
    };

    auto builder = PipelineBuilder()
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
            .withColorFormats({target.getFormat()});

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    cubemapCaptureRenderInfo = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::createIrradianceCaptureRenderInfo() {
    RenderTarget target{
        irradianceMapTexture->getImage().getMipView(ctx, 0),
        irradianceMapTexture->getFormat()
    };

    auto builder = PipelineBuilder()
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
            .withColorFormats({target.getFormat()});

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    irradianceCaptureRenderInfo = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::createPrefilterRenderInfos() {
    auto builder = PipelineBuilder()
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
                    .stageFlags = vk::ShaderStageFlagBits::eFragment,
                    .offset = 0,
                    .size = sizeof(PrefilterPushConstants),
                }
            })
            .forViews(6)
            .withColorFormats({prefilteredEnvmapTexture->getFormat()});

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    for (uint32_t i = 0; i < MAX_PREFILTER_MIP_LEVELS; i++) {
        RenderTarget target{
            prefilteredEnvmapTexture->getImage().getMipView(ctx, i),
            prefilteredEnvmapTexture->getFormat()
        };

        std::vector<RenderTarget> targets;
        targets.emplace_back(std::move(target));

        prefilterRenderInfos.emplace_back(
            builder,
            pipeline,
            std::move(targets)
        );
    }
}

void VulkanRenderer::createBrdfIntegrationRenderInfo() {
    RenderTarget target{
        brdfIntegrationMapTexture->getImage().getMipView(ctx, 0),
        brdfIntegrationMapTexture->getFormat()
    };

    auto builder = PipelineBuilder()
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
            .withColorFormats({target.getFormat()});

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    std::vector<RenderTarget> targets;
    targets.emplace_back(std::move(target));

    brdfIntegrationRenderInfo = make_unique<RenderInfo>(
        builder,
        pipeline,
        std::move(targets)
    );
}

void VulkanRenderer::createDebugQuadRenderInfos() {
    debugQuadRenderInfos.clear();

    auto builder = PipelineBuilder()
            .withVertexShader("../shaders/obj/ss-quad-vert.spv")
            .withFragmentShader("../shaders/obj/ss-quad-frag.spv")
            .withVertices<ScreenSpaceQuadVertex>()
            .withRasterizer({
                .polygonMode = vk::PolygonMode::eFill,
                .cullMode = vk::CullModeFlagBits::eNone,
                .frontFace = vk::FrontFace::eCounterClockwise,
                .lineWidth = 1.0f,
            })
            .withMultisampling({
                .rasterizationSamples = getMsaaSampleCount(),
                .minSampleShading = 1.0f,
            })
            .withDepthStencil({
                .depthTestEnable = vk::False,
                .depthWriteEnable = vk::False,
            })
            .withDescriptorLayouts({
                *debugQuadDescriptorSet->getLayout(),
            })
            .withColorFormats({swapChain->getImageFormat()})
            .withDepthFormat(swapChain->getDepthFormat());

    auto pipeline = make_shared<Pipeline>(builder.create(ctx));

    for (auto &target: swapChain->getRenderTargets(ctx)) {
        std::vector<RenderTarget> colorTargets;
        colorTargets.emplace_back(std::move(target.colorTarget));

        debugQuadRenderInfos.emplace_back(
            builder,
            pipeline,
            std::move(colorTargets),
            std::move(target.depthTarget)
        );
    }
}

// ==================== pipelines ====================

void VulkanRenderer::reloadShaders() const {
    waitIdle();

    sceneRenderInfos[0].reloadShaders(ctx);
    skyboxRenderInfos[0].reloadShaders(ctx);
    prepassRenderInfo->reloadShaders(ctx);
    ssaoRenderInfo->reloadShaders(ctx);
    cubemapCaptureRenderInfo->reloadShaders(ctx);
    irradianceCaptureRenderInfo->reloadShaders(ctx);
    prefilterRenderInfos[0].reloadShaders(ctx);
    brdfIntegrationRenderInfo->reloadShaders(ctx);
    debugQuadRenderInfos[0].reloadShaders(ctx);
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

    resultBuffer->copyFromBuffer(ctx, stagingBuffer, bufferSize);

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

    ctx.commandPool = make_unique<vk::raii::CommandPool>(*ctx.device, poolInfo);
}

void VulkanRenderer::createCommandBuffers() {
    const vk::CommandBufferAllocateInfo primaryAllocInfo{
        .commandPool = **ctx.commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(frameResources.size()),
    };

    const vk::CommandBufferAllocateInfo secondaryAllocInfo{
        .commandPool = **ctx.commandPool,
        .level = vk::CommandBufferLevel::eSecondary,
        .commandBufferCount = static_cast<uint32_t>(frameResources.size()),
    };

    vk::raii::CommandBuffers graphicsCommandBuffers{*ctx.device, primaryAllocInfo};

    vk::raii::CommandBuffers sceneCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers guiCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers prepassCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers debugCommandBuffers{*ctx.device, secondaryAllocInfo};
    vk::raii::CommandBuffers ssaoCommandBuffers{*ctx.device, secondaryAllocInfo};

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
        frameResources[i].ssaoCmdBuffer =
                {make_unique<vk::raii::CommandBuffer>(std::move(ssaoCommandBuffers[i]))};
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
        commandBuffer.beginRendering(prepassRenderInfo->get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].prepassCmdBuffer);
        commandBuffer.endRendering();
    }

    // ssao pass

    if (frameResources[currentFrameIdx].ssaoCmdBuffer.wasRecordedThisFrame) {
        commandBuffer.beginRendering(ssaoRenderInfo->get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].ssaoCmdBuffer);
        commandBuffer.endRendering();
    }

    // main pass

    if (frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame) {
        const auto &renderInfo = sceneRenderInfos[swapChain->getCurrentImageIndex()];
        commandBuffer.beginRendering(renderInfo.get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].sceneCmdBuffer);
        commandBuffer.endRendering();
    }

    // debug quad pass

    if (frameResources[currentFrameIdx].debugCmdBuffer.wasRecordedThisFrame) {
        const auto &renderInfo = sceneRenderInfos[swapChain->getCurrentImageIndex()];
        commandBuffer.beginRendering(renderInfo.get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].debugCmdBuffer);
        commandBuffer.endRendering();
    }

    // gui pass

    if (frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame) {
        const auto &renderInfo = guiRenderInfos[swapChain->getCurrentImageIndex()];
        commandBuffer.beginRendering(renderInfo.get(swapChain->getExtent(), 1, renderingFlags));
        commandBuffer.executeCommands(**frameResources[currentFrameIdx].guiCmdBuffer);
        commandBuffer.endRendering();
    }

    swapChain->transitionToPresentLayout(commandBuffer);

    commandBuffer.end();
}

// ==================== sync ====================

void VulkanRenderer::createSyncObjects() {
    const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineSemaphoreInfo{
        {},
        {
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        }
    };

    constexpr vk::SemaphoreCreateInfo binarySemaphoreInfo;

    for (auto &res: frameResources) {
        res.sync = {
            .imageAvailableSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .readyToPresentSemaphore = make_unique<vk::raii::Semaphore>(*ctx.device, binarySemaphoreInfo),
            .renderFinishedTimeline = {
                make_unique<vk::raii::Semaphore>(*ctx.device, timelineSemaphoreInfo.get<vk::SemaphoreCreateInfo>())
            },
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
        .Queue = **ctx.graphicsQueue,
        .DescriptorPool = static_cast<VkDescriptorPool>(**imguiDescriptorPool),
        .MinImageCount = imageCount,
        .ImageCount = imageCount,
        .MSAASamples = static_cast<VkSampleCountFlagBits>(getMsaaSampleCount()),
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
    }

    if (ImGui::CollapsingHeader("Renderer ", sectionFlags)) {
        // todo - convert these 2 to dynamic states
        if (ImGui::Checkbox("Cull backfaces", &cullBackFaces)) {
            queuedFrameBeginActions.emplace([&] {
                waitIdle();
                sceneRenderInfos[0].reloadShaders(ctx);
            });
        }

        if (ImGui::Checkbox("Wireframe mode", &wireframeMode)) {
            queuedFrameBeginActions.emplace([&] {
                waitIdle();
                sceneRenderInfos[0].reloadShaders(ctx);
            });
        }

        ImGui::Checkbox("SSAO", &useSsao);

        ImGui::Checkbox("IBL", &useIbl);

        static bool useMsaaDummy = useMsaa;
        if (ImGui::Checkbox("MSAA", &useMsaaDummy)) {
            queuedFrameBeginActions.emplace([this] {
                useMsaa = useMsaaDummy;

                waitIdle();
                recreateSwapChain();

                createSceneRenderInfos();
                createSkyboxRenderInfos();
                createDebugQuadRenderInfos();

                guiRenderer.reset();
                initImgui();
            });
        }

#ifndef NDEBUG
        ImGui::Separator();
        ImGui::DragFloat("Debug number", &debugNumber, 0.01, 0, std::numeric_limits<float>::max());
#endif
    }

    if (ImGui::CollapsingHeader("Lighting ", sectionFlags)) {
        ImGui::SliderFloat("Light intensity", &lightIntensity, 0.0f, 100.0f, "%.2f");
        ImGui::ColorEdit3("Light color", &lightColor.x);
        ImGui::gizmo3D("Light direction", lightDirection, 160, imguiGizmo::modeDirection);
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

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritanceInfo{
        {},
        {
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachmentFormats.size()),
            .pColorAttachmentFormats = colorAttachmentFormats.data(),
            .rasterizationSamples = getMsaaSampleCount(),
        }
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo.get<vk::CommandBufferInheritanceInfo>(),
    };

    commandBuffer.begin(beginInfo);

    guiRenderer->beginRendering();
    renderCommands();
    guiRenderer->endRendering(commandBuffer);

    commandBuffer.end();

    frameResources[currentFrameIdx].guiCmdBuffer.wasRecordedThisFrame = true;
}

bool VulkanRenderer::startFrame() {
    while (!queuedFrameBeginActions.empty()) {
        queuedFrameBeginActions.front()();
        queuedFrameBeginActions.pop();
    }

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
    frameResources[currentFrameIdx].ssaoCmdBuffer.wasRecordedThisFrame = false;
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

    const vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfo> submitInfo{
        {
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &**frameResources[currentFrameIdx].graphicsCmdBuffer,
            .signalSemaphoreCount = signalSemaphores.size(),
            .pSignalSemaphores = signalSemaphores.data(),
        },
        {
            .waitSemaphoreValueCount = static_cast<uint32_t>(waitSemaphoreValues.size()),
            .pWaitSemaphoreValues = waitSemaphoreValues.data(),
            .signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size()),
            .pSignalSemaphoreValues = signalSemaphoreValues.data(),
        }
    };

    try {
        ctx.graphicsQueue->submit(submitInfo.get<vk::SubmitInfo>());
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

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritanceInfo{
        {},
        prepassRenderInfo->getInheritanceRenderingInfo()
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo.get<vk::CommandBufferInheritanceInfo>(),
    };

    commandBuffer.begin(beginInfo);

    vkutils::cmd::setDynamicStates(commandBuffer, swapChain->getExtent());

    auto &pipeline = prepassRenderInfo->getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.bindVertexBuffers(0, **vertexBuffer, {0});
    commandBuffer.bindVertexBuffers(1, **instanceDataBuffer, {0});
    commandBuffer.bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.getLayout(),
        0,
        ***frameResources[currentFrameIdx].prepassDescriptorSet,
        nullptr
    );

    drawModel(commandBuffer, false, pipeline);

    commandBuffer.end();

    frameResources[currentFrameIdx].prepassCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::runSsaoPass() {
    if (!model || !useSsao) {
        return;
    }

    const auto &commandBuffer = *frameResources[currentFrameIdx].ssaoCmdBuffer.buffer;

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritanceInfo{
        {},
        ssaoRenderInfo->getInheritanceRenderingInfo()
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo.get<vk::CommandBufferInheritanceInfo>(),
    };

    commandBuffer.begin(beginInfo);

    vkutils::cmd::setDynamicStates(commandBuffer, swapChain->getExtent());

    auto &pipeline = ssaoRenderInfo->getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.bindVertexBuffers(0, **screenSpaceQuadVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.getLayout(),
        0,
        ***frameResources[currentFrameIdx].ssaoDescriptorSet,
        nullptr
    );

    commandBuffer.draw(screenSpaceQuadVertices.size(), 1, 0, 0);

    commandBuffer.end();

    frameResources[currentFrameIdx].ssaoCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawScene() {
    if (!model) {
        return;
    }

    const auto &commandBuffer = *frameResources[currentFrameIdx].sceneCmdBuffer.buffer;

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritanceInfo{
        {},
        sceneRenderInfos[0].getInheritanceRenderingInfo()
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo.get<vk::CommandBufferInheritanceInfo>(),
    };

    commandBuffer.begin(beginInfo);

    vkutils::cmd::setDynamicStates(commandBuffer, swapChain->getExtent());

    // skybox

    const auto &skyboxPipeline = skyboxRenderInfos[swapChain->getCurrentImageIndex()].getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **skyboxPipeline);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *skyboxPipeline.getLayout(),
        0,
        ***frameResources[currentFrameIdx].skyboxDescriptorSet,
        nullptr
    );

    commandBuffer.draw(static_cast<uint32_t>(skyboxVertices.size()), 1, 0, 0);

    // scene

    const auto &scenePipeline = sceneRenderInfos[swapChain->getCurrentImageIndex()].getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **scenePipeline);

    commandBuffer.bindVertexBuffers(0, **vertexBuffer, {0});
    commandBuffer.bindVertexBuffers(1, **instanceDataBuffer, {0});
    commandBuffer.bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *scenePipeline.getLayout(),
        0,
        {
            ***frameResources[currentFrameIdx].sceneDescriptorSet,
            ***materialsDescriptorSet,
            ***iblDescriptorSet,
        },
        nullptr
    );

    drawModel(commandBuffer, true, scenePipeline);

    commandBuffer.end();

    frameResources[currentFrameIdx].sceneCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawDebugQuad() {
    const auto &commandBuffer = *frameResources[currentFrameIdx].debugCmdBuffer.buffer;

    const vk::StructureChain<
        vk::CommandBufferInheritanceInfo,
        vk::CommandBufferInheritanceRenderingInfo
    > inheritanceInfo{
        {},
        debugQuadRenderInfos[0].getInheritanceRenderingInfo()
    };

    const vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritanceInfo.get<vk::CommandBufferInheritanceInfo>(),
    };

    commandBuffer.begin(beginInfo);

    vkutils::cmd::setDynamicStates(commandBuffer, swapChain->getExtent());

    auto &pipeline = debugQuadRenderInfos[swapChain->getCurrentImageIndex()].getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.bindVertexBuffers(0, **screenSpaceQuadVertexBuffer, {0});

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.getLayout(),
        0,
        ***debugQuadDescriptorSet,
        nullptr
    );

    commandBuffer.draw(screenSpaceQuadVertices.size(), 1, 0, 0);

    commandBuffer.end();

    frameResources[currentFrameIdx].debugCmdBuffer.wasRecordedThisFrame = true;
}

void VulkanRenderer::drawModel(const vk::raii::CommandBuffer &commandBuffer, const bool doPushConstants,
                               const Pipeline &pipeline) const {
    uint32_t indexOffset = 0;
    std::int32_t vertexOffset = 0;
    uint32_t instanceOffset = 0;

    for (const auto &mesh: model->getMeshes()) {
        // todo - make this a bit nicer (without the ugly bool)
        if (doPushConstants) {
            commandBuffer.pushConstants<ScenePushConstants>(
                *pipeline.getLayout(),
                vk::ShaderStageFlagBits::eFragment,
                0,
                ScenePushConstants{
                    .materialID = mesh.materialID
                }
            );
        }

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

    const auto commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

    vkutils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(cubemapCaptureRenderInfo->get(extent, 6));

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    const auto &pipeline = cubemapCaptureRenderInfo->getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.getLayout(),
        0,
        ***cubemapCaptureDescriptorSet,
        nullptr
    );

    commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    skyboxTexture->getImage().transitionLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        commandBuffer
    );

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);

    skyboxTexture->generateMipmaps(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void VulkanRenderer::captureIrradianceMap() const {
    const vk::Extent2D extent = irradianceMapTexture->getImage().getExtent2d();

    const auto commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

    vkutils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(irradianceCaptureRenderInfo->get(extent, 6));

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    const auto &pipeline = irradianceCaptureRenderInfo->getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *pipeline.getLayout(),
        0,
        ***envmapConvoluteDescriptorSet,
        nullptr
    );

    commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    irradianceMapTexture->getImage().transitionLayout(
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eTransferDstOptimal,
        commandBuffer
    );

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);

    irradianceMapTexture->generateMipmaps(ctx, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void VulkanRenderer::prefilterEnvmap() const {
    const auto commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

    commandBuffer.bindVertexBuffers(0, **skyboxVertexBuffer, {0});

    for (uint32_t mipLevel = 0; mipLevel < MAX_PREFILTER_MIP_LEVELS; mipLevel++) {
        const uint32_t mipScalingFactor = 1 << mipLevel;

        vk::Extent2D extent = prefilteredEnvmapTexture->getImage().getExtent2d();
        extent.width /= mipScalingFactor;
        extent.height /= mipScalingFactor;

        vkutils::cmd::setDynamicStates(commandBuffer, extent);

        commandBuffer.beginRendering(prefilterRenderInfos[mipLevel].get(extent, 6));

        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            *prefilterRenderInfos[mipLevel].getPipeline().getLayout(),
            0,
            ***envmapConvoluteDescriptorSet,
            nullptr
        );

        const auto &pipeline = prefilterRenderInfos[mipLevel].getPipeline();
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

        commandBuffer.pushConstants<PrefilterPushConstants>(
            *pipeline.getLayout(),
            vk::ShaderStageFlagBits::eFragment,
            0u,
            PrefilterPushConstants{
                .roughness = static_cast<float>(mipLevel) / (MAX_PREFILTER_MIP_LEVELS - 1)
            }
        );

        commandBuffer.draw(skyboxVertices.size(), 1, 0, 0);

        commandBuffer.endRendering();
    }

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);
}

void VulkanRenderer::computeBrdfIntegrationMap() const {
    const vk::Extent2D extent = brdfIntegrationMapTexture->getImage().getExtent2d();

    const auto commandBuffer = vkutils::cmd::beginSingleTimeCommands(ctx);

    vkutils::cmd::setDynamicStates(commandBuffer, extent);

    commandBuffer.beginRendering(brdfIntegrationRenderInfo->get(extent));

    commandBuffer.bindVertexBuffers(0, **screenSpaceQuadVertexBuffer, {0});

    const auto &pipeline = brdfIntegrationRenderInfo->getPipeline();
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **pipeline);

    commandBuffer.draw(screenSpaceQuadVertices.size(), 1, 0, 0);

    commandBuffer.endRendering();

    vkutils::cmd::endSingleTimeCommands(commandBuffer, *ctx.graphicsQueue);
}

void VulkanRenderer::updateGraphicsUniformBuffer() const {
    const glm::mat4 model = glm::translate(modelTranslate)
                            * mat4_cast(modelRotation)
                            * glm::scale(glm::vec3(modelScale));
    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 proj = camera->getProjectionMatrix();

    glm::ivec2 windowSize{};
    glfwGetWindowSize(window, &windowSize.x, &windowSize.y);

    const auto &[zNear, zFar] = camera->getClippingPlanes();

    static const glm::mat4 cubemapFaceProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);

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
            .zNear = zNear,
            .zFar = zFar,
            .useSsao = useSsao ? 1u : 0,
            .useIbl = useIbl ? 1u : 0,
            .lightIntensity = lightIntensity,
            .lightDir = glm::vec3(mat4_cast(lightDirection) * glm::vec4(-1, 0, 0, 0)),
            .lightColor = lightColor,
            .cameraPos = camera->getPos(),
        }
    };

    static const std::array cubemapFaceViews{
        glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0)),
        glm::lookAt(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)),
        glm::lookAt(glm::vec3(0), glm::vec3(0, 1, 0), glm::vec3(0, 0, -1)),
        glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1)),
        glm::lookAt(glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(0, 1, 0)),
        glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0))
    };

    for (size_t i = 0; i < 6; i++) {
        graphicsUbo.matrices.cubemapCaptureViews[i] = cubemapFaceViews[i];
    }

    memcpy(frameResources[currentFrameIdx].graphicsUboMapped, &graphicsUbo, sizeof(graphicsUbo));
}
