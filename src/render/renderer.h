#pragma once

#include <chrono>
#include <optional>
#include <vector>
#include <filesystem>
#include <array>

#include "deps/vma/vk_mem_alloc.h"

#include "libs.h"

class Mesh;
class Camera;
class Buffer;
class Texture;
class SwapChain;
class GuiRenderer;

using std::unique_ptr, std::make_unique;

static constexpr std::array validationLayers{
    "VK_LAYER_KHRONOS_validation"
};

static constexpr std::array deviceExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE2_EXTENSION_NAME,
    VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphicsComputeFamily;
    std::optional<std::uint32_t> presentFamily;

    [[nodiscard]]
    bool isComplete() const {
        return graphicsComputeFamily.has_value() && presentFamily.has_value();
    }
};

/**
 * Enumeration used to control the way the automaton's cells are colored.
 * These *have to* match the values #defined in the fragment shader.
 */
enum class ColoringPreset : std::uint32_t {
    COORD_RGB = 0,
    STATE_GRADIENT = 1,
    DISTANCE_GRADIENT = 2,
    SOLID_COLOR = 3,
};

/**
 * Information held in the fragment shader's uniform buffer.
 * This (obviously) has to exactly match the corresponding definition in the fragment shader.
 */
struct GraphicsUBO {
    struct WindowRes {
        std::uint32_t windowWidth;
        std::uint32_t windowHeight;
    };

    struct Matrices {
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 inverseVp;
        glm::mat4 staticView;
    };

    struct MiscData {
        glm::vec3 camera_pos;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) MiscData misc{};
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
    VmaAllocator get() const { return allocator; }
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

    vk::raii::Context vkCtx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<vk::raii::Queue> graphicsQueue;
    unique_ptr<vk::raii::Queue> presentQueue;

    unique_ptr<SwapChain> swapChain;

    unique_ptr<Mesh> mesh;

    unique_ptr<Texture> texture;
    unique_ptr<Texture> skyboxTexture;

    unique_ptr<vk::raii::DescriptorPool> descriptorPool;

    unique_ptr<vk::raii::RenderPass> renderPass;

    unique_ptr<vk::raii::DescriptorSetLayout> sceneGraphicsSetLayout;
    unique_ptr<vk::raii::DescriptorSetLayout> skyboxGraphicsSetLayout;
    unique_ptr<vk::raii::PipelineLayout> scenePipelineLayout;
    unique_ptr<vk::raii::PipelineLayout> skyboxPipelineLayout;
    unique_ptr<vk::raii::Pipeline> scenePipeline;
    unique_ptr<vk::raii::Pipeline> skyboxPipeline;

    unique_ptr<vk::raii::CommandPool> commandPool;

    unique_ptr<Buffer> vertexBuffer;
    unique_ptr<Buffer> indexBuffer;

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

        unique_ptr<vk::raii::CommandBuffer> graphicsCmdBuf, guiCmdBuf;

        unique_ptr<Buffer> graphicsUniformBuffer;
        void *graphicsUboMapped{};

        unique_ptr<vk::raii::DescriptorSet> sceneDescriptorSet;
        unique_ptr<vk::raii::DescriptorSet> skyboxDescriptorSet;

    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frameResources;

    vk::SampleCountFlagBits msaaSampleCount = vk::SampleCountFlagBits::e1;

    unique_ptr<vk::raii::DescriptorPool> imguiDescriptorPool;
    unique_ptr<GuiRenderer> guiRenderer;

    // miscellaneous state variables

    bool doShowGui = false;

    std::uint32_t currentFrameIdx = 0;

    bool framebufferResized = false;

    glm::vec3 backgroundColor = glm::vec3(26, 26, 26) / 255.0f;

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
    GuiRenderer& getGuiRenderer() const { return *guiRenderer; }

    void tick(float deltaTime);

    /**
     * Waits until the device has completed all previously submitted commands.
     */
    void waitIdle() const { ctx.device->waitIdle(); }

    /**
     * Locks or unlocks the cursor. When the cursor is locked, it's confined to the center
     * of the screen and camera rotates according to its movement. When it's unlocked, it's
     * visible and free to move around the screen; most importantly able to use the GUI.
     */
    void setIsCursorLocked(bool b) const;

    void loadModel(const std::filesystem::path& meshPath, const std::filesystem::path& texturePath);

private:
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

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

    static bool checkDeviceSubgroupSupport(const vk::raii::PhysicalDevice &physicalDevice);

    // ==================== logical device ====================

    void createLogicalDevice();

    // ==================== assets ====================

    void createSkyboxTexture();

    // ==================== swap chain ====================

    void recreateSwapChain();

    // ==================== descriptors ====================

    void createDescriptorSetLayouts();

    void createSceneDescriptorSetLayouts();

    void createSkyboxDescriptorSetLayouts();

    void createDescriptorPool();

    void createDescriptorSets();

    void createSceneDescriptorSets();

    void createSkyboxDescriptorSets();

    // ==================== graphics pipeline ====================

    void createRenderPass();

    void createPipelines();

    void createScenePipeline();

    void createSkyboxPipeline();

    [[nodiscard]]
    vk::raii::ShaderModule createShaderModule(const std::filesystem::path &path) const;

    // ==================== multisampling ====================

    [[nodiscard]]
    vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    // ==================== buffers ====================

    void createVertexBuffer();

    void createIndexBuffer();

    template<typename ElemType>
    unique_ptr<Buffer> createLocalBuffer(const std::vector<ElemType> &contents, vk::BufferUsageFlags usage);

    void createUniformBuffers();

    // ==================== commands ====================

    void createCommandPool();

    void createCommandBuffers();

    void recordGraphicsCommandBuffer();

    // ==================== sync ====================

    void createSyncObjects();

    // ==================== gui ====================

    void initImgui();

public:
    void renderGuiSection();

    void setDoShowGui(const bool b) { doShowGui = b; }

    // ==================== render loop ====================

    void startFrame();

    void endFrame();

    void renderGui(const std::function<void()> &renderCommands) const;

    void drawScene();

private:
    void updateGraphicsUniformBuffer() const;
};
