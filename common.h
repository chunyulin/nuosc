#pragma once

//==== Start of global flags
#define NFLAVOR 2
#define DIM 3
#if DIM != 3
#error Only DIM=3 is supported
#endif

#define WENO7
#define BC_PERI
#define KO_ORD_3
//#define IM_V2D_ICOSAHEDRA
//#define ADV_TEST

#define PROFILE
#define DEBUG
#define COSENU_MPI
//#define GDR_OFF
//#define SYNC_NCCL
//#define SYNC_COPY
//#define SYNC_MPI_SENDRECV
//#define SYNC_MPI_ONESIDE_COPY

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

#include <tuple>
#include <stack>

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

#define _SIMD_ simd
#if defined(INTEL_COMPILER)
  #define RESTRICT restrict
#else
  #define RESTRICT __restrict
#endif

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


#ifdef SYNC_NCCL
#include "nccl.h"
#define NCCLCHECK(cmd) do {     \
    ncclResult_t res = cmd;     \
    if (res != ncclSuccess) {   \
        printf("Failed, NCCL error %s:%d '%s'\n", __FILE__,__LINE__,ncclGetErrorString(res)); \
        exit(EXIT_FAILURE);     \
    }                           \
    } while(0)
#endif
