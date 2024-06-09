#pragma once

#include "deps/vma/vk_mem_alloc.h"
#include "src/render/libs.h"

/**
 * Abstraction over a Vulkan buffer, making it easier to manage by hiding all the Vulkan API calls.
 * These buffers are allocated using VMA and are currently suited mostly for two scenarios: first,
 * when one needs a device-local buffer, and second, when one needs a host-visible and host-coherent
 * buffer, e.g. for use as a staging buffer.
 */
class Buffer {
    VmaAllocator allocator{};
    vk::Buffer buffer;
    VmaAllocation allocation{};
    void *mapped = nullptr;

public:
    explicit Buffer(VmaAllocator _allocator, vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties);

    ~Buffer();

    Buffer(const Buffer &other) = delete;
    Buffer(Buffer &&other) = delete;
    Buffer &operator=(const Buffer &other) = delete;
    Buffer &operator=(Buffer &&other) = delete;

    /**
     * Returns a raw handle to the actual Vulkan buffer.
     *
     * @return Handle to the buffer.
     */
    [[nodiscard]]
    const vk::Buffer& get() const { return buffer; }

    /**
     * Maps the buffer's memory to host memory. This requires the buffer to *not* be created
     * with the vk::MemoryPropertyFlagBits::eDeviceLocal flag set in `properties` during object creation.
     * If already mapped, just returns the pointer to the previous mapping.
     *
     * @return Pointer to the mapped memory.
     */
    [[nodiscard]]
    void *map();

    /**
     * Unmaps the memory, after which the pointer returned by `map()` becomes invalidated.
     * Fails if `map()` wasn't called beforehand.
     */
    void unmap();
};
