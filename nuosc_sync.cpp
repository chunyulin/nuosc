#include "nuosc_class.h"

void NuOsc::packSend(const FieldVar* v0) {

    #ifdef PROFILING_BREAKDOWNS
    auto t0 = std::chrono::high_resolution_clock::now();
    #endif
    pack_buffer(v0);
    #ifdef PROFILING_BREAKDOWNS
    auto t1 = std::chrono::high_resolution_clock::now();
    #endif

    sync_launch();

    #ifdef PROFILING_BREAKDOWNS
    auto t2 = std::chrono::high_resolution_clock::now();
    t_packing += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0 ).count();
    t_sync += std::chrono::duration_cast<std::chrono::milliseconds>( t2-t1 ).count();
    #endif
}

void NuOsc::waitall() {
#ifdef NVTX
    nvtxRangePush("Wait and unpack");
#endif
#ifdef COSENU_MPI
    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    #if defined(SYNC_NCCL)
    for (int i=0;i<2*DIM;++i)  cudaStreamCreate(&stream[i]);
    //cudaDeviceSynchronize();
    #elif defined(SYNC_MPI_ONESIDE_COPY)
    for (int d=0;d<DIM;++d)    MPI_Win_fence(0, w_pb[d]);
    #elif defined(SYNC_MPI_SENDRECV)
    #elif defined(SYNC_COPY)
    #else // default for nonblocking
    const int n_reqs = DIM*4;
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    #endif

    #if defined (_OPENACC) && defined (GDR_OFF) && !defined (SYNC_NCCL)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        real *pbuf = &(pb[d][0]);
        #pragma acc update device( pbuf[(2*npb):(4*npb)] ) async
    }
    #pragma acc wait
    #endif
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::sync_launch() {
#ifdef NVTX
    nvtxRangePush("Sync launch");
#endif
#ifdef COSENU_MPI
    ulong nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    #pragma omp parallel for num_threads(DIM)
    #if defined (_OPENACC) && !defined(SYNC_NCCL)
    for (int d=0;d<DIM;++d) {
        real *pbuf = &(pb[d][0]);
        #ifdef GDR_OFF
            const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
            #pragma acc update self( pbuf[0:2*npb] ) async
        #else
            #pragma acc host_data use_device(pbuf)
        #endif
    }
    #pragma acc wait
    #endif

    #if defined(SYNC_MPI_ONESIDE_COPY)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        MPI_Win_fence(0, w_pb[d]);
        MPI_Put(&pb[d][0],   1, t_pb[d], nb[0][0], 2 /* skip old block */, 1, t_pb[d], w_pb[d]);
        MPI_Put(&pb[d][npb], 1, t_pb[d], nb[0][1], 3 /* skip old block */, 1, t_pb[d], w_pb[d]);
    }
    #elif defined(SYNC_MPI_SENDRECV)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        MPI_Sendrecv(&pb[d][    0], 1, t_pb[d], nb[d][0],   d*2, &pb[d][2*npb], 1, t_pb[d], nb[d][1],   d*2, CartCOMM, MPI_STATUS_IGNORE );
        MPI_Sendrecv(&pb[d][  npb], 1, t_pb[d], nb[d][1], 1+d*2, &pb[d][3*npb], 1, t_pb[d], nb[d][0], 1+d*2, CartCOMM, MPI_STATUS_IGNORE );
    }
    #elif defined(SYNC_NCCL)
    NCCLCHECK( ncclGroupStart() );
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        NCCLCHECK( ncclSend(&pb[0][    0], npb, ncclDouble, nb[d][0], _ncclcomm, stream[2*d]) );
        NCCLCHECK( ncclRecv(&pb[0][2*npb], npb, ncclDouble, nb[d][1], _ncclcomm, stream[2*d]) );
        NCCLCHECK( ncclSend(&pb[0][  npb], npb, ncclDouble, nb[d][1], _ncclcomm, stream[2*d+1]) );
        NCCLCHECK( ncclRecv(&pb[0][3*npb], npb, ncclDouble, nb[d][0], _ncclcomm, stream[2*d+1]) );
    }
    NCCLCHECK( ncclGroupEnd() );
    #elif defined(SYNC_COPY)
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        memcpy(&pb[d][2*npb], &pb[d][0], 2*npb*sizeof(real));
    }
    #else
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZ/nx[d]*gx[d];   // total size of halo
        MPI_Irecv(&pb[d][2*npb], 1, t_pb[d], nb[d][1],   d*2, CartCOMM, &reqs[d*4]);
        MPI_Irecv(&pb[d][3*npb], 1, t_pb[d], nb[d][0], 1+d*2, CartCOMM, &reqs[d*4+1]);
        MPI_Isend(&pb[d][    0], 1, t_pb[d], nb[d][0],   d*2, CartCOMM, &reqs[d*4+2]);
        MPI_Isend(&pb[d][  npb], 1, t_pb[d], nb[d][1], 1+d*2, CartCOMM, &reqs[d*4+3]);
    }
    #endif
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}
