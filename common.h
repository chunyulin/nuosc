#pragma once

//==== Start of global flags
#define NFLAVOR 2
#define DIM 3
#if DIM != 3
#error Only DIM=3 is supported
#endif

#define COSENU_MPI
#define WENO7
//#define GDR_OFF
//#define PROFILING
//#define SYNC_COPY
//#define SYNC_MPI_SENDRECV
//#define SYNC_MPI_ONESIDE_COPY
//#define ADV_TEST
#define BC_PERI
#define KO_ORD_3

//==== End of global flags


#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <list>
#include <algorithm>
#include <string>

using std::cout;
using std::endl;
using std::cin;
using std::string;

using std::sqrt;
using std::abs;
using std::max;
using std::min;
using std::cos;
using std::sin;


typedef double real;
typedef std::vector<double> Vec;

#ifdef PAPI
#include <papi.h>
#endif


#ifdef NVTX
#include <nvToolsExt.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif
#include <omp.h>

// We adopot domain decomposition over spacial coordinates (x,z), not over v to avoid data exchange for v-integral.
// Halo size is 16* nv* (nx*gz + nz*gx).
#ifdef COSENU_MPI
#include <mpi.h>
#endif
