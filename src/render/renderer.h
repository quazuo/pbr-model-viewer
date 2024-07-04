#pragma once

#include <optional>
#include <vector>
#include <filesystem>
#include <array>

#include "deps/vma/vk_mem_alloc.h"

#include "libs.h"
#include "globals.h"

class Image;
class InputManager;
class Model;
class Camera;
class Buffer;
class PipelinePack;
class RenderPass;
class DescriptorSet;
class Texture;
class SwapChain;
class GuiRenderer;

static constexpr std::array validationLayers{
    "VK_LAYER_KHRONOS_validation"
};

static constexpr std::array deviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE2_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsComputeFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const {
        return graphicsComputeFamily.has_value() && presentFamily.has_value();
    }
};

/**
 * Information held in the fragment shader's uniform buffer.
 * This (obviously) has to exactly match the corresponding definition in the fragment shader.
 */
struct GraphicsUBO {
    struct WindowRes {
        uint32_t windowWidth;
        uint32_t windowHeight;
    };

    struct Matrices {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 inverseVp;
        glm::mat4 staticView;
        glm::mat4 cubemapCaptureProj;
    };

    struct MiscData {
        float debugNumber;
        glm::vec3 cameraPos;
        glm::vec3 lightDir;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
};

struct SkyboxPushConstants {
    glm::mat4 view;
};

struct PrefilterPushConstants {
    glm::mat4 view;
    float roughness;
};

/**
 * Simple RAII-preserving wrapper class for the VMA allocator.
 */
class VmaAllocatorWrapper {
    VmaAllocator allocator{};

public:
    VmaAllocatorWrapper(vk::PhysicalDevice physicalDevice, vk::Device device, vk::Instance instance);

    ~VmaAllocatorWrapper();

    VmaAllocatorWrapper(const VmaAllocatorWrapper &other) = delete;

    VmaAllocatorWrapper(VmaAllocatorWrapper &&other) = delete;

    VmaAllocatorWrapper &operator=(const VmaAllocatorWrapper &other) = delete;

    VmaAllocatorWrapper &operator=(VmaAllocatorWrapper &&other) = delete;

    [[nodiscard]] VmaAllocator operator*() const { return allocator; }
};

/**
 * Helper structure used to pass handles to essential Vulkan objects which are used while interacting with the API.
 * Introduced so that we can preserve top-down data flow and no object needs to refer to a renderer object
 * to get access to these.
 */
struct RendererContext {
    unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    unique_ptr<vk::raii::Device> device;
    unique_ptr<VmaAllocatorWrapper> allocator;
};

class VulkanRenderer {
    struct GLFWwindow *window = nullptr;

    unique_ptr<Camera> camera;

    unique_ptr<InputManager> inputManager;

    vk::raii::Context vkCtx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<vk::raii::Queue> graphicsQueue;
    unique_ptr<vk::raii::Queue> presentQueue;

    unique_ptr<SwapChain> swapChain;

    unique_ptr<Model> model;

    unique_ptr<Texture> albedoTexture;
    unique_ptr<Texture> normalTexture;
    unique_ptr<Texture> ormTexture;

    unique_ptr<Texture> skyboxTexture;
    unique_ptr<Texture> envmapTexture;
    unique_ptr<Texture> irradianceMapTexture;
    unique_ptr<Texture> prefilteredEnvmapTexture;
    unique_ptr<Texture> brdfIntegrationMapTexture;

    unique_ptr<vk::raii::DescriptorPool> descriptorPool;

    unique_ptr<RenderPass> sceneRenderPass;
    unique_ptr<RenderPass> cubemapCaptureRenderPass;
    unique_ptr<RenderPass> envmapConvoluteRenderPass;
    unique_ptr<RenderPass> brdfIntegrationRenderPass;

    unique_ptr<vk::raii::DescriptorSetLayout> sceneDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> skyboxDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> cubemapCaptureDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> envmapConvoluteDescriptorLayout;

    unique_ptr<vk::raii::Framebuffer> cubemapCaptureFramebuffer;
    unique_ptr<vk::raii::Framebuffer> irradianceCaptureFramebuffer;
    std::vector<unique_ptr<vk::raii::Framebuffer>> prefilterFramebuffers;
    unique_ptr<vk::raii::Framebuffer> brdfIntegrationFramebuffer;

    unique_ptr<DescriptorSet> cubemapCaptureDescriptorSet;
    unique_ptr<DescriptorSet> envmapConvoluteDescriptorSet;

    unique_ptr<PipelinePack> scenePipeline;
    unique_ptr<PipelinePack> skyboxPipeline;
    unique_ptr<PipelinePack> cubemapCapturePipelines;
    unique_ptr<PipelinePack> irradianceCapturePipelines;
    unique_ptr<PipelinePack> prefilterPipelines;
    unique_ptr<PipelinePack> brdfIntegrationPipeline;

    unique_ptr<vk::raii::CommandPool> commandPool;

    unique_ptr<Buffer> vertexBuffer;
    unique_ptr<Buffer> indexBuffer;
    unique_ptr<Buffer> instanceDataBuffer;
    unique_ptr<Buffer> skyboxVertexBuffer;
    unique_ptr<Buffer> screenSpaceQuadVertexBuffer;

    struct FrameResources {
        struct {
            struct Timeline {
                unique_ptr<vk::raii::Semaphore> semaphore;
                std::uint64_t timeline = 0u;
            };

            unique_ptr<vk::raii::Semaphore> imageAvailableSemaphore;
            unique_ptr<vk::raii::Semaphore> readyToPresentSemaphore;
            Timeline renderFinishedTimeline;
        } sync;

        // primary command buffer
        unique_ptr<vk::raii::CommandBuffer> graphicsCmdBuffer;

        struct SecondaryCommandBuffer {
            unique_ptr<vk::raii::CommandBuffer> buffer;
            bool wasRecordedThisFrame = false;
        } sceneCmdBuffer, guiCmdBuffer;

        unique_ptr<Buffer> graphicsUniformBuffer;
        void *graphicsUboMapped{};

        unique_ptr<DescriptorSet> sceneDescriptorSet;
        unique_ptr<DescriptorSet> skyboxDescriptorSet;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frameResources;

    vk::SampleCountFlagBits msaaSampleCount = vk::SampleCountFlagBits::e1;

    static constexpr vk::Extent2D cubemapExtent = {2048, 2048};
    static constexpr vk::Extent2D brdfIntegrationMapExtent = { 512, 512 };

    static constexpr auto hdrEnvmapFormat = vk::Format::eR32G32B32A32Sfloat;
    static constexpr auto brdfIntegrationMapFormat = vk::Format::eR8G8B8A8Unorm;

    static constexpr uint32_t maxPrefilterMipLevels = 5;

    unique_ptr<vk::raii::DescriptorPool> imguiDescriptorPool;
    unique_ptr<GuiRenderer> guiRenderer;

    // miscellaneous state variables

    uint32_t currentFrameIdx = 0;

    bool framebufferResized = false;

    glm::vec3 backgroundColor = glm::vec3(26, 26, 26) / 255.0f;

    float modelScale = 1.0f;
    glm::vec3 modelTranslate{};
    glm::quat modelRotation{1, 0, 0, 0};

    float debugNumber = 0;

    bool cullBackFaces = false;
    bool wireframeMode = false;

public:
    explicit VulkanRenderer();

    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer &other) = delete;

    VulkanRenderer(VulkanRenderer &&other) = delete;

    VulkanRenderer &operator=(const VulkanRenderer &other) = delete;

    VulkanRenderer &operator=(VulkanRenderer &&other) = delete;

    [[nodiscard]] GLFWwindow *getWindow() const { return window; }

    [[nodiscard]] GuiRenderer &getGuiRenderer() const { return *guiRenderer; }

    void tick(float deltaTime);

    /**
     * Waits until the device has completed all previously submitted commands.
     */
    void waitIdle() const { ctx.device->waitIdle(); }

    void loadModel(const std::filesystem::path &path);

    void loadAlbedoTexture(const std::filesystem::path &path);

    void loadNormalMap(const std::filesystem::path &path);

    void loadOrmMap(const std::filesystem::path &path);

    void loadOrmMap(const std::filesystem::path &aoPath, const std::filesystem::path &roughnessPath,
                    const std::filesystem::path &metallicPath);

    void loadRmaMap(const std::filesystem::path &path);

    void loadEnvironmentMap(const std::filesystem::path &path);

private:
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

    void bindMouseDragActions();

    // ==================== instance creation ====================

    void createInstance();

    static std::vector<const char *> getRequiredExtensions();

    // ==================== validation layers ====================

    static bool checkValidationLayerSupport();

    static vk::DebugUtilsMessengerCreateInfoEXT makeDebugMessengerCreateInfo();

    void setupDebugMessenger();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData
    );

    // ==================== window surface ====================

    void createSurface();

    // ==================== physical device ====================

    void pickPhysicalDevice();

    [[nodiscard]] bool isDeviceSuitable(const vk::raii::PhysicalDevice &physicalDevice) const;

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice &physicalDevice) const;

    static bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice &physicalDevice);

    // ==================== logical device ====================

    void createLogicalDevice();

    // ==================== assets ====================

    void createIblTextures();

    // ==================== swap chain ====================

    void recreateSwapChain();

    // ==================== descriptors ====================

    void createDescriptorSetLayouts();

    void createSceneDescriptorSetLayout();

    void createSkyboxDescriptorSetLayout();

    void createCubemapCaptureDescriptorSetLayout();

    void createEnvmapConvoluteDescriptorSetLayout();

    void createDescriptorPool();

    void createSceneDescriptorSets();

    void createSkyboxDescriptorSets();

    void createCubemapCaptureDescriptorSets();

    void createEnvmapConvoluteDescriptorSets();

    // ==================== render passes ====================

    void createRenderPass();

    void createCubemapCaptureRenderPass();

    void createCubemapConvoluteRenderPass();

    void createBrdfIntegrationRenderPass();

    // ==================== pipelines ====================

    void createScenePipeline();

    void createSkyboxPipeline();

    void createCubemapCapturePipeline();

    void createIrradianceCapturePipeline();

    void createPrefilterPipeline();

    void createBrdfIntegrationPipeline();

    // ==================== multisampling ====================

    [[nodiscard]] vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    // ==================== buffers ====================

    void createVertexBuffer();

    void createSkyboxVertexBuffer();

    void createScreenSpaceQuadVertexBuffer();

    void createIndexBuffer();

    template<typename ElemType>
    unique_ptr<Buffer> createLocalBuffer(const std::vector<ElemType> &contents, vk::BufferUsageFlags usage);

    void createUniformBuffers();

    // ==================== framebuffers ====================

    void createCubemapCaptureFramebuffer();

    void createIrradianceCaptureFramebuffer();

    void createPrefilterFramebuffers();

    void createBrdfIntegrationFramebuffer();

    [[nodiscard]] unique_ptr<vk::raii::Framebuffer>
    createPerLayerCubemapFramebuffer(const Texture &texture, const vk::raii::RenderPass &renderPass) const;

    [[nodiscard]] unique_ptr<vk::raii::Framebuffer>
    createMipPerLayerCubemapFramebuffer(const Texture &texture, const vk::raii::RenderPass &renderPass,
                                           uint32_t mipLevel) const;

    // ==================== commands ====================

    void createCommandPool();

    void createCommandBuffers();

    void recordGraphicsCommandBuffer() const;

    // ==================== sync ====================

    void createSyncObjects();

    // ==================== gui ====================

    void initImgui();

public:
    void renderGuiSection();

    // ==================== render loop ====================

    bool startFrame();

    void endFrame();

    void renderGui(const std::function<void()> &renderCommands);

    void drawScene();

    void captureCubemap();

    void captureIrradianceMap();

    void prefilterEnvmap();

    void computeBrdfIntegrationMap();

private:
    void updateGraphicsUniformBuffer() const;
};
