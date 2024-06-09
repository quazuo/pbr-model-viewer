#pragma once

#include <chrono>

class Stopwatch {
    float period;
    std::chrono::high_resolution_clock::time_point startTime, lastTime, currentTime;

public:
    explicit Stopwatch(const float p = 0.0f) : period(p) { reset(); }

    void reset();

    void tick();

    [[nodiscard]]
    float getElapsed() const { return std::chrono::duration<float>(currentTime - startTime).count(); }

    [[nodiscard]]
    float getDelta() const { return std::chrono::duration<float>(currentTime - lastTime).count(); }

    [[nodiscard]]
    bool isFinished() const { return getElapsed() >= period; }
};
