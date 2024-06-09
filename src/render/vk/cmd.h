#pragma once

#include "src/render/libs.h"

namespace utils::cmd {
    /**
     * Allocates and begins a new command buffer which is supposed to be recorded once
     * and destroyed after submission.
     *
     * @param device Logical device handle.
     * @param commandPool Command pool from which to allocate.
     * @return The created single-use command buffer.
     */
    [[nodiscard]] vk::raii::CommandBuffer
    beginSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool);

    /**
     * Ends a single-time command buffer created beforehand by `beginSingleTimeCommands`.
     * The buffer is then submitted and execution stops until the commands are fully processed.
     *
     * @param commandBuffer The single-use command buffer which should be ended.
     * @param queue The queue to which the buffer should be submitted.
     */
    void endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer, const vk::raii::Queue &queue);

    /**
     * Convenience wrapper over `beginSingleTimeCommands` and `endSingleTimeCommands`.
     *
     * @param device Logical device handle.
     * @param commandPool Command pool from which to allocate.
     * @param queue The queue to which the buffer should be submitted.
     * @param func Lambda containing commands with which the command buffer will be filled.
     */
    void doSingleTimeCommands(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool,
                              const vk::raii::Queue &queue,
                              const std::function<void(const vk::raii::CommandBuffer &)> &func);
}
