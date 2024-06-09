#include "buffer.h"

Buffer::Buffer(const VmaAllocator _allocator, const vk::DeviceSize size, const vk::BufferUsageFlags usage,
               const vk::MemoryPropertyFlags properties)
    : allocator(_allocator) {
    const vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateFlags flags;
    if (properties & vk::MemoryPropertyFlagBits::eDeviceLocal) {
        flags = 0;
    } else {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }

    const VmaAllocationCreateInfo allocInfo{
        .flags = flags,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(properties)
    };

    const auto result = vmaCreateBuffer(
        allocator,
        reinterpret_cast<const VkBufferCreateInfo *>(&bufferInfo),
        &allocInfo,
        reinterpret_cast<VkBuffer *>(&buffer),
        &allocation,
        nullptr
    );

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer!");
    }
}

Buffer::~Buffer() {
    if (mapped) {
        unmap();
    }

    vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);
}

void *Buffer::map() {
    if (!mapped && vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("failed to map buffer memory!");
    }

    return mapped;
}

void Buffer::unmap() {
    if (!mapped) {
        throw std::runtime_error("tried to unmap a buffer that wasn't mapped!");
    }

    vmaUnmapMemory(allocator, allocation);
    mapped = nullptr;
}
