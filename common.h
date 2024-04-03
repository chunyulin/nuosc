#pragma once
//==== Start of global flags
#define NFLAVOR 2
#define DIM 3
#if DIM != 3
#error Only DIM=3 is supported
#endif

#define COSENU_MPI
//#define SCHEME_FD8
//#define SCHEME_WENO7
//#define PROFILE 20
//#define VERBOSE
//#define SYNC_NCCL
//#define SYNC_COPY
//#define SYNC_MPI_SENDRECV
//#define SYNC_MPI_ONESIDE_COPY
#define BC_PERI
#define KO_ORD_3
//#define ADV_TEST
//#define IM_V2D_ICOSAHEDRA
//#define NOT_OVERLAP
//#define GDR_OFF

#define WALLTIME_LIMIT_HOUR 3.9
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

const string CKPT="./ckpt";
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
