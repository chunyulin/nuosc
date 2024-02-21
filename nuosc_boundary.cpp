#include "nuosc_class.h"

void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { 
    cout << "Not implemented." << endl;
    assert(0);
}

void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("Periodic Boundary");
#endif
    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

    #pragma acc parallel loop collapse(4)
    #pragma omp parallel for simd collapse(4)
    for (int i=0;i<gx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<nx[2]; ++k)
    for (int v=0;v<nv;    ++v) {
        //x lower side of ghost <-- x upper
        auto l0 = idx( -i-1,j,k,v );
        auto l1 = idx( nx[0]-i-1,j,k,v );
        #pragma unroll
        for(int f=0;f<nvar;++f)  in->wf[f][l0] = in->wf[f][l1];
        //x upper side of ghost <-- x lower
        auto r0 = idx( nx[0]+i,j,k,v );
        auto r1 = idx( i,j,k,v );
        #pragma unroll
        for(int f=0;f<nvar;++f)   in->wf[f][r0] = in->wf[f][r1];
    }

    #pragma acc parallel loop collapse(4)
    #pragma omp parallel for simd collapse(4)
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<gx[1]; ++j)
    for (int k=0;k<nx[2]; ++k)
    for (int v=0;v<nv;    ++v) {
        // y lower side of ghost <-- y upper
        auto l0 = idx(i,-j-1,k,v);
        auto l1 = idx(i,nx[1]-j-1,k,v);
        #pragma unroll
        for(int f=0;f<nvar;++f)  in->wf[f][l0] = in->wf[f][l1];
        //y upper side of ghost <-- y lower
        auto r0 = idx(i,nx[1]+j,k,v);
        auto r1 = idx(i,j,k,v);
        #pragma unroll
        for(int f=0;f<nvar;++f)   in->wf[f][r0] = in->wf[f][r1];
    }

    #pragma acc parallel loop collapse(4)
    #pragma omp parallel for simd collapse(4)
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<gx[2]; ++k)
    for (int v=0;v<nv;    ++v) {
        // z lower side of ghost <-- z upper
        auto l0 = idx(i,j,-k-1,v);
        auto l1 = idx(i,j,nx[2]-k-1,v);
        #pragma unroll
        for(int f=0;f<nvar;++f)  in->wf[f][l0] = in->wf[f][l1];
        //z upper side of ghost <-- z lower
        auto r0 = idx(i,j,nx[2]+k,v);
        auto r1 = idx(i,j,k,v);
        #pragma unroll
        for(int f=0;f<nvar;++f)   in->wf[f][r0] = in->wf[f][r1];
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

#if DIM==2
 #define PBIX(i,j,k,v) ( (i*nz + k)*nv+v )*nvar;
 #define PBIZ(i,j,k,v) ( (i*gz + k)*nv+v )*nvar;
 #define NPBX nvar*gx*nz*nv
 #define NPBZ nvar*nx*gz*nv
#elif DIM == 3
 #define PBIX(i,j,k,v) nvar*( v + nv*( k + nx[2]*( j + nx[1]*i ) ) );
 #define PBIY(i,j,k,v) nvar*( v + nv*( k + nx[2]*( j + gx[1]*i ) ) );
 #define PBIZ(i,j,k,v) nvar*( v + nv*( k + gx[2]*( j + nx[1]*i ) ) );
 #define NPBX gx[0]*nx[1]*nx[2]*nv*nvar
 #define NPBY nx[0]*gx[1]*nx[2]*nv*nvar
 #define NPBZ nx[0]*nx[1]*gx[2]*nv*nvar
#endif
void NuOsc::pack_buffer(const FieldVar* in) {
#ifdef NVTX
    nvtxRangePush("Pack Buffer");
#endif
    { // X lower side
        real *pbuf = &(pb[0][0]);
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<gx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r1 = idx(i,j,k,v);
            auto tid8 = PBIX(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][r1];
        }
    }
    {  // X upper side
        real *pbuf = &(pb[0][NPBX]);
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<gx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l1 = idx(nx[0]-i-1,j,k,v);
            auto tid8 = PBIX(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][l1];
        }
    }
    { // Y lower side
        real *pbuf = &(pb[1][0]);
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<gx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r1 = idx(i,j,k,v);
            auto tid8 = PBIY(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][r1];
        }
    }
    {  // Y upper side
        real *pbuf = &(pb[1][NPBY]);
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<gx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l1 = idx(i,nx[1]-j-1,k,v);
            auto tid8 = PBIY(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][l1];
        }
    }
    { // Pack Z lower
        real *pbuf = &(pb[2][0]);    // THINK: OpenACC error w/o this!!
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<gx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r1 = idx(i,j,k,v);
            auto tid8 = PBIZ(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][r1];
        }
    }
    { // Pack Z upper
        real *pbuf = &(pb[2][NPBZ]);
        #pragma acc parallel loop collapse(4) independent async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<gx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l1 = idx(i,j,nx[2]-k-1,v);
            auto tid8 = PBIZ(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f)  pbuf[tid8+f] = in->wf[f][l1];
        }
    }
    #pragma acc wait
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::unpack_buffer(FieldVar* out) {
#ifdef NVTX
    nvtxRangePush("Unpack Buffer");
#endif
    { // recovery X upper halo from the neighbor lower side
        real *pbuf = &(pb[0][2*NPBX]);    // THINK: OpenACC error w/o this (ie., offset inside pragma)!!
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<gx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r0 = idx(nx[0]+i,j,k,v);
            auto tid8 = PBIX(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][r0] = pbuf[tid8+f];
        }
    }
    { // recovery X lower halo from the neighbor upper side
        real *pbuf = &(pb[0][3*NPBX]);
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<gx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l0 = idx(-i-1,j,k,v);
            auto tid8 = PBIX(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][l0] = pbuf[tid8+f];
        }
    }
    { // recovery Y upper halo from the neighbor lower side
        real *pbuf = &(pb[1][2*NPBY]);
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<gx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r0 = idx(i,nx[1]+j,k,v);
            auto tid8 = PBIY(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][r0] = pbuf[tid8+f];
        }
    }
    { // recovery Y lower halo from the neighbor upper side
        real *pbuf = &(pb[1][3*NPBY]);
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<gx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l0 = idx(i,-j-1,k,v);
            auto tid8 = PBIY(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][l0] = pbuf[tid8+f];
        }
    }
    { // Z lower side
        real *pbuf = &(pb[2][2*NPBZ]);
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<gx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto r0 = idx(i,j,nx[2]+k,v);
            auto tid8 = PBIZ(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][r0] = pbuf[tid8+f];
        }
    }
    { // Z upper side
        real *pbuf = &(pb[2][3*NPBZ]);
        #pragma acc parallel loop collapse(4) async
        #pragma omp parallel for simd collapse(4)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<gx[2]; ++k)
        for (int v=0;v<nv;    ++v) {
            auto l0 = idx(i,j,-k-1,v);
            auto tid8 = PBIZ(i,j,k,v);
            #pragma unroll
            for(int f=0;f<nvar;++f) out->wf[f][l0] = pbuf[tid8+f];
        }
    }
    #pragma acc wait
#ifdef NVTX
    nvtxRangePop();
#endif
}