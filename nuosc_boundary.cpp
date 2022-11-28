#include "nuosc_class.h"


void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("PeriodicBoundary");
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
        for (int v=0;v<grid.nv; ++v) {
            // z lower side of ghost <-- z upper
            auto l0 = grid.idx(i,-j-1,v);
            auto l1 = grid.idx(i,grid.nz-j-1,v);
            in->ee    [l0] = in->ee    [l1];
            in->xx    [l0] = in->xx    [l1];
            in->ex_re [l0] = in->ex_re [l1];
            in->ex_im [l0] = in->ex_im [l1];
            in->bee   [l0] = in->bee   [l1];
            in->bxx   [l0] = in->bxx   [l1];
            in->bex_re[l0] = in->bex_re[l1];
            in->bex_im[l0] = in->bex_im[l1];
            //z upper side of ghost <-- z lower
            auto r0 = grid.idx(i,grid.nz+j,v);
            auto r1 = grid.idx(i,j,v);
            in->ee    [r0] = in->ee    [r1];
            in->xx    [r0] = in->xx    [r1];
            in->ex_re [r0] = in->ex_re [r1];
            in->ex_im [r0] = in->ex_im [r1];
            in->bee   [r0] = in->bee   [r1];
            in->bxx   [r0] = in->bxx   [r1];
            in->bex_re[r0] = in->bex_re[r1];
            in->bex_im[r0] = in->bex_im[r1];
        }


#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
    for (int i=0;i<grid.gx; ++i)
        for (int j=0;j<grid.nz; ++j)
            for (int v=0;v<grid.nv; ++v) {
                //x lower side of ghost <-- x upper
                auto l0 = grid.idx(-i-1,j,v);
                auto l1 = grid.idx(grid.nx-i-1,j,v);
                in->ee    [l0] = in->ee    [l1];
                in->xx    [l0] = in->xx    [l1];
                in->ex_re [l0] = in->ex_re [l1];
                in->ex_im [l0] = in->ex_im [l1];
                in->bee   [l0] = in->bee   [l1];
                in->bxx   [l0] = in->bxx   [l1];
                in->bex_re[l0] = in->bex_re[l1];
                in->bex_im[l0] = in->bex_im[l1];
                //x upper side of ghost <-- x lower
                auto r0 = grid.idx(grid.nx+i,j,v);
                auto r1 = grid.idx(i,j,v);
                in->ee    [r0] = in->ee    [r1];
                in->xx    [r0] = in->xx    [r1];
                in->ex_re [r0] = in->ex_re [r1];
                in->ex_im [r0] = in->ex_im [r1];
                in->bee   [r0] = in->bee   [r1];
                in->bxx   [r0] = in->bxx   [r1];
                in->bex_re[r0] = in->bex_re[r1];
                in->bex_im[r0] = in->bex_im[r1];
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
void NuOsc::pack_buffer(const FieldVar* in) {

    { // X lower side
    real *pbuf = &(grid.pbX[0]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r1 = grid.idx(i,j,v);
        auto tid8 = PBIX(i,j,v);
        pbuf[tid8+0] = in->ee    [r1];
        pbuf[tid8+1] = in->xx    [r1];
        pbuf[tid8+2] = in->ex_re [r1];
        pbuf[tid8+3] = in->ex_im [r1];
        pbuf[tid8+4] = in->bee   [r1];
        pbuf[tid8+5] = in->bxx   [r1];
        pbuf[tid8+6] = in->bex_re[r1];
        pbuf[tid8+7] = in->bex_im[r1];
    }
    }

    {  // X upper side
    real *pbuf = &(grid.pbX[NPBX]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) independent async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l1 = grid.idx(grid.nx-i-1,j,v);
        auto tid8 = PBIX(i,j,v);
        pbuf[tid8+0] = in->ee    [l1];
        pbuf[tid8+1] = in->xx    [l1];
        pbuf[tid8+2] = in->ex_re [l1];
        pbuf[tid8+3] = in->ex_im [l1];
        pbuf[tid8+4] = in->bee   [l1];
        pbuf[tid8+5] = in->bxx   [l1];
        pbuf[tid8+6] = in->bex_re[l1];
        pbuf[tid8+7] = in->bex_im[l1];
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
        pbuf[tid8+0] = in->ee    [r1];
        pbuf[tid8+1] = in->xx    [r1];
        pbuf[tid8+2] = in->ex_re [r1];
        pbuf[tid8+3] = in->ex_im [r1];
        pbuf[tid8+4] = in->bee   [r1];
        pbuf[tid8+5] = in->bxx   [r1];
        pbuf[tid8+6] = in->bex_re[r1];
        pbuf[tid8+7] = in->bex_im[r1];
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
        pbuf[tid8+0] = in->ee    [l1];
        pbuf[tid8+1] = in->xx    [l1];
        pbuf[tid8+2] = in->ex_re [l1];
        pbuf[tid8+3] = in->ex_im [l1];
        pbuf[tid8+4] = in->bee   [l1];
        pbuf[tid8+5] = in->bxx   [l1];
        pbuf[tid8+6] = in->bex_re[l1];
        pbuf[tid8+7] = in->bex_im[l1];
    }
    }

    #pragma acc wait
}

void NuOsc::unpack_buffer(FieldVar* out) {

    { // recovery X upper halo from the neighbor lower side
    real *pbuf = &(grid.pbX[2*NPBX]);    // THINK: OpenACC error w/o this (ie., offset inside pragma)!!
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r0 = grid.idx(grid.nx+i,j,v);
        auto tid8 = PBIX(i,j,v);
        out->ee    [r0] = pbuf[tid8+0];
        out->xx    [r0] = pbuf[tid8+1];
        out->ex_re [r0] = pbuf[tid8+2];
        out->ex_im [r0] = pbuf[tid8+3];
        out->bee   [r0] = pbuf[tid8+4];
        out->bxx   [r0] = pbuf[tid8+5];
        out->bex_re[r0] = pbuf[tid8+6];
        out->bex_im[r0] = pbuf[tid8+7];
    }
    }

    { // recovery X lower halo from the neighbor upper side
    real *pbuf = &(grid.pbX[3*NPBX]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) async
    for (int i=0;i<grid.gx; ++i)
    for (int j=0;j<grid.nz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l0 = grid.idx(-i-1,j,v);
        auto tid8 = PBIX(i,j,v);
        out->ee    [l0] = pbuf[tid8+0];
        out->xx    [l0] = pbuf[tid8+1];
        out->ex_re [l0] = pbuf[tid8+2];
        out->ex_im [l0] = pbuf[tid8+3];
        out->bee   [l0] = pbuf[tid8+4];
        out->bxx   [l0] = pbuf[tid8+5];
        out->bex_re[l0] = pbuf[tid8+6];
        out->bex_im[l0] = pbuf[tid8+7];
    }
    }

    { // Z lower side
    real *pbuf = &(grid.pbZ[2*NPBZ]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto r0 = grid.idx(i,grid.nz+j,v);
        auto tid8 = PBIZ(i,j,v);
        out->ee    [r0] = pbuf[tid8+0];
        out->xx    [r0] = pbuf[tid8+1];
        out->ex_re [r0] = pbuf[tid8+2];
        out->ex_im [r0] = pbuf[tid8+3];
        out->bee   [r0] = pbuf[tid8+4];
        out->bxx   [r0] = pbuf[tid8+5];
        out->bex_re[r0] = pbuf[tid8+6];
        out->bex_im[r0] = pbuf[tid8+7];
    }
    }

    { // Z upper side
    real *pbuf = &(grid.pbZ[3*NPBZ]);
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3) async
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.gz; ++j)
    for (int v=0;v<grid.nv; ++v) {
        auto l0 = grid.idx(i,-j-1,v);
        auto tid8 = PBIZ(i,j,v);
        out->ee    [l0] = pbuf[tid8+0];
        out->xx    [l0] = pbuf[tid8+1];
        out->ex_re [l0] = pbuf[tid8+2];
        out->ex_im [l0] = pbuf[tid8+3];
        out->bee   [l0] = pbuf[tid8+4];
        out->bxx   [l0] = pbuf[tid8+5];
        out->bex_re[l0] = pbuf[tid8+6];
        out->bex_im[l0] = pbuf[tid8+7];
    }
    }
    #pragma acc wait
}


void NuOsc::sync_boundary(FieldVar* v0) {

    pack_buffer(v0);
#if defined(SYNC_COPY)
    grid.sync_buffer_copy();
#elif defined(SYNC_MPI_ONESIDE_COPY)
    grid.sync_buffer();
#else
    grid.sync_buffer_isend();
#endif
    unpack_buffer(v0);


}


