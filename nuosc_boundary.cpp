#include "nuosc_class.h"


void NuOsc::updatePeriodicBoundary(FieldVar * const __restrict in) {
#ifdef NVTX
    nvtxRangePush("PeriodicBoundary");
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]
    const std::vector<real*> fvars = in->getAllFields();

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3) independent
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        { // z lower side of ghost <-- z upper
        auto l0 = grid.idx(i,-j-1,v);
        auto l1 = grid.idx(i,grid.nz-j-1,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fvars.at(f)[l0] = fvars.at(f)[l1];
        } {
        //z upper side of ghost <-- z lower
        auto r0 = grid.idx(i,grid.nz+j,v);
        auto r1 = grid.idx(i,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fvars.at(f)[r0] = fvars.at(f)[r1];
        }
    }

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3) independent
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        { //x lower side of ghost <-- x upper
        auto l0 = grid.idx(-i-1,j,v);
        auto l1 = grid.idx(grid.nx-i-1,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fvars.at(f)[l0] = fvars.at(f)[l1];
        } { //x upper side of ghost <-- x lower
        auto r0 = grid.idx(grid.nx+i,j,v);
        auto r1 = grid.idx(i,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fvars.at(f)[r0] = fvars.at(f)[r1];
        }
    }

#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { 
    cout << "Not implemented." << endl;
    assert(0);
}


#define PBIX(i,j,v) ( (i*grid.nz + j)*grid.nv+v )*nvar;
#define PBIZ(i,j,v) ( (i*grid.gz + j)*grid.nv+v )*nvar;
#define NPBX nvar*grid.gx*grid.nz*grid.nv
#define NPBZ nvar*grid.nx*grid.gz*grid.nv
void NuOsc::pack_buffer(FieldVar* const in) {

    std::vector<real*> fvars = in->getAllFields();

    { // X lower side
    real * const pbuf = &(grid.pbX[0]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r1 = grid.idx(i,j,v);
        auto tid8 = PBIX(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  pbuf[tid8+f] = fvars.at(f)[r1];
    }
    }

    {  // X upper side
    real * const pbuf = &(grid.pbX[NPBX]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l1 = grid.idx(grid.nx-i-1,j,v);
        auto tid8 = PBIX(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  pbuf[tid8+f] = fvars.at(f)[l1];
    }
    }

    { // Pack Z lower
    real *pbuf = &(grid.pbZ[0]);    // THINK: OpenACC error w/o this!!
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r1 = grid.idx(i,j,v);
        auto tid8 = PBIZ(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  pbuf[tid8+f] = fvars.at(f)[r1];
    }
    }

    { // Pack Z upper
    real *pbuf = &(grid.pbZ[NPBZ]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l1 = grid.idx(i,grid.nz-j-1,v);
        auto tid8 = PBIZ(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  pbuf[tid8+f] = fvars.at(f)[l1];
    }
    }

    #pragma acc wait
}

void NuOsc::unpack_buffer(FieldVar* out) {

    std::vector<real*> fvars = out->getAllFields();

    { // recovery X upper halo from the neighbor lower side
    real *pbuf = &(grid.pbX[2*NPBX]);    // THINK: OpenACC error w/o this (ie., offset inside pragma)!!
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r0 = grid.idx(grid.nx+i,j,v);
        auto tid8 = PBIX(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  fvars.at(f)[r0] = pbuf[tid8+f];
    }
    }

    { // recovery X lower halo from the neighbor upper side
    real *pbuf = &(grid.pbX[3*NPBX]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l0 = grid.idx(-i-1,j,v);
        auto tid8 = PBIX(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  fvars.at(f)[l0] = pbuf[tid8+f];
    }
    }

    { // Z lower side
    real *pbuf = &(grid.pbZ[2*NPBZ]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r0 = grid.idx(i,grid.nz+j,v);
        auto tid8 = PBIZ(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  fvars.at(f)[r0] = pbuf[tid8+f];
    }
    }

    { // Z upper side
    real *pbuf = &(grid.pbZ[3*NPBZ]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l0 = grid.idx(i,-j-1,v);
        auto tid8 = PBIZ(i,j,v);
        #pragma unroll
        for (int f=0;f<nvar;++f)  fvars.at(f)[l0] = pbuf[tid8+f];
    }
    }
    #pragma acc wait
}


void NuOsc::sync_boundary(FieldVar* v0) {

    pack_buffer(v0);
    #if   defined(SYNC_NCCL)
    grid.sync_nccl();
    #elif defined(SYNC_COPY)
    grid.sync_copy();    //  Only for test in one node.
    #elif defined(SYNC_MPI_ONESIDE_COPY)
    grid.sync_put();     //  Not work on T2?
    #elif defined(SYNC_MPI_SENDRECV)
    grid.sync_sendrecv();
    #else
    grid.sync_isend();
    #endif

    unpack_buffer(v0);


}
