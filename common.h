#pragma once

//==== Global flag
#define KO_ORD_3
#define BC_PERI

#define COSENU_MPI
//#define GDR_OFF
//#define SYNC_MPI_ONESIDE_COPY


//#define ADV_TEST



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

