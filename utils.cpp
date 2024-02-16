#include "utils.h"

namespace utils {

void printMemoryUsage() {
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  std::cout << "Max resident memory: " << r.ru_maxrss/1024.0/1024.0 << " GB" << std::endl;
}
double getMemoryUsage() {
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  return r.ru_maxrss;
}

// Simple timer for single node
void reset_timer() { _t0 = std::chrono::high_resolution_clock::now(); };
float msecs_since() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-_t0).count(); };

}