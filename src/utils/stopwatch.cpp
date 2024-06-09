#include "stopwatch.h"

void Stopwatch::reset() {
    startTime = std::chrono::high_resolution_clock::now();
    lastTime = startTime;
    currentTime = startTime;
}

void Stopwatch::tick() {
    lastTime = currentTime;
    currentTime = std::chrono::high_resolution_clock::now();
}
