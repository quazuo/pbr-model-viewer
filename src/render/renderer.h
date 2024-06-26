#pragma once

#include <optional>
#include <vector>
#include <filesystem>
#include <array>

#include "deps/vma/vk_mem_alloc.h"

#include "libs.h"
#include "globals.h"
#include "vk/pipeline.h"

class Image;
class InputManager;
class Model;
class Camera;
class Buffer;
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

    [[nodiscard]]
    bool isComplete() const {
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
    };

    struct MiscData {
        uint32_t useIBL;
        glm::vec3 cameraPos;
        glm::vec3 lightDir;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
};

struct CubemapCapturePushConstants {
    glm::mat4 view;
    glm::mat4 proj;
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

    [[nodiscard]]
    VmaAllocator operator*() const { return allocator; }
};

/**
 * Helper structure used to handles to essential Vulkan objects which are used while interacting with the API.
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

    unique_ptr<vk::raii::DescriptorPool> descriptorPool;

    unique_ptr<vk::raii::RenderPass> renderPass;

    unique_ptr<vk::raii::DescriptorSetLayout> sceneDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> skyboxDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> cubemapCaptureDescriptorLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> irradianceCaptureDescriptorLayout;

    unique_ptr<Pipeline> scenePipeline;
    unique_ptr<Pipeline> skyboxPipeline;
    std::vector<unique_ptr<Pipeline>> cubemapCapturePipelines;
    std::vector<unique_ptr<Pipeline>> irradianceCapturePipelines;

    unique_ptr<vk::raii::CommandPool> commandPool;

    unique_ptr<Buffer> vertexBuffer;
    unique_ptr<Buffer> indexBuffer;
    unique_ptr<Buffer> instanceDataBuffer;
    unique_ptr<Buffer> skyboxVertexBuffer;

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

        unique_ptr<vk::raii::DescriptorSet> sceneDescriptorSet;
        unique_ptr<vk::raii::DescriptorSet> skyboxDescriptorSet;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frameResources;

    vk::Extent2D cubemapExtent = {2048u, 2048u};

    struct CaptureResources {
        unique_ptr<vk::raii::RenderPass> renderPass;
        unique_ptr<vk::raii::Framebuffer> framebuffer;
        unique_ptr<vk::raii::DescriptorSet> descriptorSet;
    } cubemapCaptureResources, irradianceCaptureResources;

    vk::SampleCountFlagBits msaaSampleCount = vk::SampleCountFlagBits::e1;

    unique_ptr<vk::raii::DescriptorPool> imguiDescriptorPool;
    unique_ptr<GuiRenderer> guiRenderer;

    // miscellaneous state variables

    uint32_t currentFrameIdx = 0;

    bool framebufferResized = false;

    glm::vec3 backgroundColor = glm::vec3(26, 26, 26) / 255.0f;

    float modelScale = 1.0f;
    glm::vec3 modelTranslate{};

    bool useIBL = true;

public:
    explicit VulkanRenderer();

    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer &other) = delete;

    VulkanRenderer(VulkanRenderer &&other) = delete;

    VulkanRenderer &operator=(const VulkanRenderer &other) = delete;

    VulkanRenderer &operator=(VulkanRenderer &&other) = delete;

    [[nodiscard]]
    GLFWwindow *getWindow() const { return window; }

    [[nodiscard]]
    GuiRenderer &getGuiRenderer() const { return *guiRenderer; }

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

    void buildDescriptors();

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

    [[nodiscard]]
    bool isDeviceSuitable(const vk::raii::PhysicalDevice &physicalDevice) const;

    [[nodiscard]]
    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice &physicalDevice) const;

    static bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice &physicalDevice);

    // ==================== logical device ====================

    void createLogicalDevice();

    // ==================== assets ====================

    void createSkyboxTextures();

    // ==================== swap chain ====================

    void recreateSwapChain();

    // ==================== descriptors ====================

    void createDescriptorSetLayouts();

    void createSceneDescriptorSetLayouts();

    void createSkyboxDescriptorSetLayouts();

    void createCubemapCaptureDescriptorSetLayouts();

    void createIrradianceCaptureDescriptorSetLayouts();

    void createDescriptorPool();

    void createSceneDescriptorSets();

    void createSkyboxDescriptorSets();

    void createCubemapCaptureDescriptorSets();

    void createIrradianceCaptureDescriptorSets();

    // ==================== graphics pipeline ====================

    void createRenderPass();

    void createCubemapCaptureRenderPass();

    void createIrradianceCaptureRenderPass();

    void createPipelines();

    void createScenePipeline();

    void createSkyboxPipeline();

    void createCubemapCapturePipeline();

    void createIrradianceCapturePipeline();

    [[nodiscard]]
    vk::raii::ShaderModule createShaderModule(const std::filesystem::path &path) const;

    // ==================== multisampling ====================

    [[nodiscard]]
    vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    // ==================== skybox ====================

    void createSkyboxResources();

    // ==================== buffers ====================

    void createVertexBuffer();

    void createIndexBuffer();

    template<typename ElemType>
    unique_ptr<Buffer> createLocalBuffer(const std::vector<ElemType> &contents, vk::BufferUsageFlags usage);

    void createUniformBuffers();

    // ==================== framebuffers ====================

    void createCubemapCaptureFramebuffer();

    void createIrradianceCaptureFramebuffer();

    [[nodiscard]]
    unique_ptr<vk::raii::Framebuffer> createPerLayerCubemapFramebuffer(const Texture &texture,
                                                                       const vk::raii::RenderPass &renderPass) const;

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

private:
    void updateGraphicsUniformBuffer() const;
};
