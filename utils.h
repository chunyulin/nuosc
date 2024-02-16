#pragma once

#include <chrono>
#include <iostream>
#include <sys/resource.h>

namespace utils {

void printMemoryUsage();
double getMemoryUsage();

// Simple timer for single node
static std::chrono::time_point<std::chrono::high_resolution_clock> _t0 = std::chrono::high_resolution_clock::now();
void reset_timer();
float msecs_since();
}