#include "cmd.h"

vk::raii::CommandBuffer
utils::cmd::beginSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool) {
    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1U,
    };

    vk::raii::CommandBuffers commandBuffers{device, allocInfo};
    vk::raii::CommandBuffer buffer{std::move(commandBuffers[0])};

    constexpr vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    buffer.begin(beginInfo);

    return buffer;
}

void utils::cmd::endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer, const vk::raii::Queue &queue) {
    commandBuffer.end();

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1U,
        .pCommandBuffers = &*commandBuffer,
    };

    queue.submit(submitInfo);
    queue.waitIdle();
}

void utils::cmd::doSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool,
                                      const vk::raii::Queue &queue,
                                      const std::function<void(const vk::raii::CommandBuffer &)> &func) {
    const vk::raii::CommandBuffer cmdBuffer = beginSingleTimeCommands(device, commandPool);
    func(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, queue);
}
