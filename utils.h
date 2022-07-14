#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>
#include <iostream>
#include <sys/resource.h>
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

#endif
