#include "nuosc_class.h"

void NuOsc::sync_boundary(FieldVar* v0) {

    #ifdef PROFILING
    auto t0 = std::chrono::high_resolution_clock::now();
    #endif
    pack_buffer(v0);

    #ifdef PROFILING
    auto t1 = std::chrono::high_resolution_clock::now();
    #endif
#if defined(SYNC_COPY)
    sync_copy();
#elif defined(SYNC_MPI_ONESIDE_COPY)
    sync_put();
#elif defined(SYNC_MPI_SENDRECV)
    sync_sendrecv();
#elif defined(SYNC_MPI_ISENDRECV)
    sync_isendrecv();
#else
    sync_isend();
#endif
    #ifdef PROFILING
    auto t2 = std::chrono::high_resolution_clock::now();
    #endif
    unpack_buffer(v0);

    #ifdef PROFILING
    t_packing += std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now()-t2+t1-t0 ).count();
    t_sync += std::chrono::duration_cast<std::chrono::milliseconds>( t2-t1 ).count();
    #endif
}



void NuOsc::sync_put() {
#ifdef NVTX
    nvtxRangePush("Sync Put");
#endif
#ifdef COSENU_MPI
    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #ifdef GDR_OFF
            #pragma acc update self( pb[d][0:2*npb] )
        #else
            #pragma acc host_data use_device(pb[d])
        #endif
        {
            MPI_Win_fence(0, w_pb[d]);
            MPI_Put(&pb[d][0],   1, t_pb[d], nb[0][0], 2 /* skip old block */, 1, t_pb[d], w_pb[d]);
            MPI_Put(&pb[d][npb], 1, t_pb[d], nb[0][1], 3 /* skip old block */, 1, t_pb[d], w_pb[d]);
            MPI_Win_fence(0, w_pb[d]);
        }
        #ifdef GDR_OFF
        #pragma acc update device( pb[d][0:4*npb] )
        #endif
    }
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::sync_sendrecv() {
#ifdef NVTX
    nvtxRangePush("Sync SendRecv");
#endif
#ifdef COSENU_MPI
    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #ifdef GDR_OFF
          #pragma acc update self( pb[d][0:2*npb] )
        #else
          #pragma acc host_data use_device(pb[d])
        #endif
        {
            MPI_Sendrecv(&pb[d][    0], 1, t_pb[d], nb[d][0],   d*2, &pb[d][2*npb], 1, t_pb[d], nb[d][1],   d*2, CartCOMM, MPI_STATUS_IGNORE );
            MPI_Sendrecv(&pb[d][  npb], 1, t_pb[d], nb[d][1], 1+d*2, &pb[d][3*npb], 1, t_pb[d], nb[d][0], 1+d*2, CartCOMM, MPI_STATUS_IGNORE );
        }
    }
    #if defined (_OPENACC) && defined (GDR_OFF)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #pragma acc update device( pb[d][(2*npb):(4*npb)] )
    }
    #endif
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::sync_isend() {
#ifdef NVTX
    nvtxRangePush("Sync Isend");
#endif
#ifdef COSENU_MPI
    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    const int n_reqs = DIM*4;
    MPI_Request reqs[n_reqs];
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #ifdef GDR_OFF
          #pragma acc update self( pb[d][0:2*npb] )
        #else
          #pragma acc host_data use_device(pb[d])
        #endif
        {
            MPI_Irecv(&pb[d][2*npb], 1, t_pb[d], nb[d][1],   d*2, CartCOMM, &reqs[d*4]);
            MPI_Irecv(&pb[d][3*npb], 1, t_pb[d], nb[d][0], 1+d*2, CartCOMM, &reqs[d*4+1]);
            MPI_Isend(&pb[d][    0], 1, t_pb[d], nb[d][0],   d*2, CartCOMM, &reqs[d*4+2]);
            MPI_Isend(&pb[d][  npb], 1, t_pb[d], nb[d][1], 1+d*2, CartCOMM, &reqs[d*4+3]);
        }
    }
    #ifdef DEBUG
    MPI_Status stats[n_reqs];
    if (MPI_SUCCESS != MPI_Waitall(n_reqs, reqs, stats) ) {
        for (int i=0; i<n_reqs; ++i) {
         printf("MPI_ERROR: Rank %d sent to %d of %d.\n", stats[i].MPI_SOURCE, myrank, stats[i].MPI_TAG );
        }
    }
    #else
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    #endif

    #if defined (_OPENACC) && defined (GDR_OFF)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #pragma acc update device( pb[d][(2*npb):(4*npb)] )
    }
    #endif
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::sync_copy() {   // Only for single-node test

    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        memcpy(&pb[d][2*npb], &pb[d][0], 2*npb*sizeof(real));
    }
}

