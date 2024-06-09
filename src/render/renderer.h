#pragma once

#include <chrono>
#include <optional>
#include <vector>
#include <filesystem>
#include <array>

#include "deps/vma/vk_mem_alloc.h"

#include "src/automaton.h"
#include "libs.h"
#include "camera.h"
#include "vk/buffer.h"
#include "vk/swapchain.h"
#include "gui/gui.h"
#include "src/utils/octree-gen.h"

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
    };

    struct ColoringData {
        std::uint32_t doNeighborShading;
        ColoringPreset coloringPreset;
        glm::vec3 color1;
        glm::vec3 color2;
        glm::vec3 backgroundColor;
    };

    struct MiscData {
        float fogDistance;
        glm::vec3 cameraPos;
    };

    struct AutomatonInfo {
        std::uint32_t gridDepth;
        std::uint32_t stateCount;
    };

    alignas(16) WindowRes window{};
    alignas(16) Matrices matrices{};
    alignas(16) ColoringData coloring{};
    alignas(16) MiscData misc{};
    alignas(16) AutomatonInfo automaton{};
};

/**
 * Information held in the compute shader's uniform buffer.
 * This (obviously) has to exactly match the corresponding definition in the compute shader.
 */
struct ComputeUBO {
    AutomatonConfig config;
};

/**
 * Information sent to the compute shader via push constants.
 * This is used primarily because we dispatch more than once per frame and each dispatch
 * does different things, which push constants let us control easily.
 */
struct ComputePushConstants {
    std::uint32_t level{};
    std::uint32_t pyramidHeight{};
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

class AutomatonRenderer {
    struct GLFWwindow *window = nullptr;

    unique_ptr<Camera> camera;

    vk::raii::Context vkCtx;
    unique_ptr<vk::raii::Instance> instance;
    unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    unique_ptr<vk::raii::SurfaceKHR> surface;

    RendererContext ctx;

    unique_ptr<vk::raii::Queue> graphicsQueue;
    unique_ptr<vk::raii::Queue> computeQueue;
    unique_ptr<vk::raii::Queue> presentQueue;

    unique_ptr<SwapChain> swapChain;

    unique_ptr<vk::raii::DescriptorPool> descriptorPool;

    unique_ptr<vk::raii::RenderPass> renderPass;

    /**
     * We use two graphics descriptor sets for each frame, as there are two kinds of information we need
     * to provide to fragment shaders:
     *
     * 1. per-frame uniform buffer with stuff like MVP matrices and other info,
     * 2. handles to storage buffers containing the most recent state of the automaton.
     *
     * As the handles change on a non-per-frame basis, we provide them in a different descriptor set
     * which we swap whenever the automaton's state updates.
     */
    struct GraphicsLayouts {
        unique_ptr<vk::raii::DescriptorSetLayout> frameSetLayout;
        unique_ptr<vk::raii::DescriptorSetLayout> ssboSetLayout;
    } graphicsDescriptorSetLayouts;
    unique_ptr<vk::raii::PipelineLayout> graphicsPipelineLayout;
    unique_ptr<vk::raii::Pipeline> graphicsPipeline;

    unique_ptr<vk::raii::DescriptorSetLayout> computeDescriptorSetLayout;
    unique_ptr<vk::raii::PipelineLayout> computePipelineLayout;
    unique_ptr<vk::raii::Pipeline> computePipeline;

    unique_ptr<vk::raii::CommandPool> commandPool;

    unique_ptr<Buffer> vertexBuffer;
    unique_ptr<Buffer> indexBuffer;

    struct FrameResources {
        unique_ptr<vk::raii::CommandBuffer> graphicsCmdBuf, guiCmdBuf, computeCmdBuf;

        struct {
            struct Timeline {
                unique_ptr<vk::raii::Semaphore> semaphore;
                std::uint64_t timeline = 0u;
            };

            unique_ptr<vk::raii::Semaphore> imageAvailableSemaphore;
            unique_ptr<vk::raii::Semaphore> readyToPresentSemaphore;
            Timeline renderFinishedTimeline;
            Timeline computeFinishedTimeline;
        } sync;

        unique_ptr<Buffer> graphicsUniformBuffer;


        void *graphicsUboMapped{};

        unique_ptr<vk::raii::DescriptorSet> graphicsDescriptorSet;
    };

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frameResources;

    struct AutomatonResources {
        unique_ptr<Buffer> shaderStorageBuffer;
        unique_ptr<vk::raii::DescriptorSet> graphicsDescriptorSet;
        unique_ptr<vk::raii::DescriptorSet> computeDescriptorSet;

        unique_ptr<Buffer> computeUniformBuffer;
        void *computeUboMapped{};
    };

    /**
     * This is pretty much a set-in-stone constant and should not change.
     * It has to be at least 2 as we need two separate storage buffers to correctly mutate the automaton's state.
     * It also shouldn't be more than 2 since it would introduce unnecessary memory bloat.
     */
    static constexpr size_t AUTOMATON_RESOURCE_COUNT = 2;
    std::array<AutomatonResources, AUTOMATON_RESOURCE_COUNT> automatonResources;

    vk::SampleCountFlagBits msaaSampleCount = vk::SampleCountFlagBits::e1;

    unique_ptr<vk::raii::DescriptorPool> imguiDescriptorPool;
    unique_ptr<GuiRenderer> guiRenderer;

    // miscellaneous state variables

    bool doShowGui = false;

    AutomatonConfig automatonConfig;

    std::uint32_t currentFrameIdx = 0;
    std::uint32_t mostRecentSsboIdx = 0;

    bool framebufferResized = false;

    static constexpr glm::uvec3 WORK_GROUP_SIZE = {8, 8, 8};

    bool usePyramidAcceleration = true;

    float fogDistance = 50.0f;
    bool doNeighborShading = true;
    ColoringPreset coloringPreset = ColoringPreset::COORD_RGB;
    glm::vec3 cellColor1 = glm::vec3(252, 70, 107) / 255.0f;
    glm::vec3 cellColor2 = glm::vec3(63, 94, 251) / 255.0f;
    glm::vec3 backgroundColor = glm::vec3(26, 26, 26) / 255.0f;

public:
    explicit AutomatonRenderer(const AutomatonConfig &config);

    ~AutomatonRenderer();

    AutomatonRenderer(const AutomatonRenderer &other) = delete;
    AutomatonRenderer(AutomatonRenderer &&other) = delete;
    AutomatonRenderer &operator=(const AutomatonRenderer &other) = delete;
    AutomatonRenderer &operator=(AutomatonRenderer &&other) = delete;

    [[nodiscard]] GLFWwindow *getWindow() const { return window; }

    void tick(float deltaTime);

    /**
     * Waits until the device has completed all previously submitted commands.
     */
    void waitIdle() const { ctx.device->waitIdle(); }

    void updateAutomatonConfig(const AutomatonConfig &config);

    /**
     * Locks or unlocks the cursor. When the cursor is locked, it's confined to the center
     * of the screen and camera rotates according to its movement. When it's unlocked, it's
     * visible and free to move around the screen; most importantly able to use the GUI.
     */
    void setIsCursorLocked(bool b) const;

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

    // ==================== swap chain ====================

    void recreateSwapChain();

    // ==================== descriptors ====================

    void createDescriptorSetLayouts();

    void createGraphicsDescriptorSetLayouts();

    void createComputeDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSets();

    void createGraphicsDescriptorSets();

    void createComputeDescriptorSets();

    // ==================== graphics pipeline ====================

    void createRenderPass();

    void createGraphicsPipeline();

    void createComputePipeline();

    [[nodiscard]]
    vk::raii::ShaderModule createShaderModule(const std::filesystem::path &path) const;

    // ==================== multisampling ====================

    [[nodiscard]]
    vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    // ==================== buffers ====================

    void createVertexBuffer();

    void createIndexBuffer();

    void createUniformBuffers();

    void createShaderStorageBuffers();

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) const;

public:
    void fillSsbos(const OctreeGen::OctreeBuf &initValues) const;

    void rebuildSsbos();

private:
    // ==================== commands ====================

    void createCommandPool();

    void createCommandBuffers();

    void recordGraphicsCommandBuffer();

    void recordComputeCommandBuffer();

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

    void runCompute();

private:
    void updateGraphicsUniformBuffer() const;

    void updateComputeUniformBuffers() const;
};
