#include "nuosc_class.h"
#include "utils.h"

void NuOsc::calRHS(FieldVar * __restrict out, FieldVar * __restrict in) {

    #ifdef COSENU_MPI
//#define PROFILING
#ifdef PROFILING
if (!myrank) utils::reset_timer();
#endif
    packSend(in);
#ifdef PROFILING
if (!myrank) printf("[%.4f] PackSend.\n", utils::msecs_since());
#endif
    {
    int bb[] = {gx[0],nx[0]-gx[0], gx[1],nx[1]-gx[1], gx[2],nx[2]-gx[2]};
    calRHS_core(out, in, bb);
    }
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Interior RHS done.\n", utils::msecs_since());
#endif
    waitall();
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Waitall.\n", utils::msecs_since());
#endif
    unpack_buffer(in);
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Unpack.\n", utils::msecs_since());
#endif

#ifdef PROFILING
if (!myrank) utils::reset_timer();
#endif
    { // Even slower if using omp parallel sections here!
    int bb[] = {0,gx[0],           gx[1],nx[1]-gx[1], gx[2],nx[2]-gx[2]};
    calRHS_core(out, in, bb);
    }{
    int bb[] = {nx[0]-gx[0],nx[0], gx[1],nx[1]-gx[1], gx[2],nx[2]-gx[2]};
    calRHS_core(out, in, bb);
    }{
    int bb[] = {0,nx[0], 0,gx[1],           gx[2],nx[2]-gx[2]};
    calRHS_core(out, in, bb);
    }{
    int bb[] = {0,nx[0], nx[1]-gx[1],nx[1], gx[2],nx[2]-gx[2]};
    calRHS_core(out, in, bb);
    }{
    int bb[] = {0,nx[0], 0,nx[1], 0,gx[2]};
    calRHS_core(out, in, bb);
    }{
    int bb[] = {0,nx[0], 0,nx[1], nx[2]-gx[2],nx[2]};
    calRHS_core(out, in, bb);
    }
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Boundary RHS done.\n", utils::msecs_since());
#endif

    #else
    updatePeriodicBoundary(in);
    int bb[] = {0,nx[0], 0,nx[1], 0,nx[2]};
    calRHS_core(out, in, bb);
    #endif
}

void NuOsc::calRHS_core(FieldVar * __restrict out, const FieldVar * __restrict in, const int bb[2*DIM]) {
#ifdef NVTX
    nvtxRangePush("calRHS");
#endif

    // for abbreviation
    const int nzv  = nv* (nx[2] + 2*gx[2]);
    const int nyzv = nzv*(nx[1] + 2*gx[1]);
// gang worker num_workers(64)
#pragma acc parallel loop independent collapse(3)
#pragma omp parallel for collapse(3)
    for (int i=bb[0];i<bb[1]; ++i)
    for (int j=bb[2];j<bb[3]; ++j)
    for (int k=bb[4];k<bb[5]; ++k) {

        // common integral over v'
        real idv_bexR_m_exR  = 0;
        real idv_bexI_p_exI  = 0;
        real idv_bxx_m_bee_m_xx_p_ee  = 0;
        real ivxdv_bexR_m_exR = 0;
        real ivxdv_bexI_p_exI = 0;
        real ivxdv_bxx_m_bee_m_xx_p_ee = 0;
        real ivydv_bexR_m_exR = 0;
        real ivydv_bexI_p_exI = 0;
        real ivydv_bxx_m_bee_m_xx_p_ee = 0;
        real ivzdv_bexR_m_exR = 0;
        real ivzdv_bexI_p_exI = 0;
        real ivzdv_bxx_m_bee_m_xx_p_ee = 0;

        // Common integral factor over spatial points
        #pragma acc loop reduction(+:idv_bexR_m_exR,idv_bexI_p_exI,idv_bxx_m_bee_m_xx_p_ee, ivxdv_bexR_m_exR,ivxdv_bexI_p_exI,ivxdv_bxx_m_bee_m_xx_p_ee, ivydv_bexR_m_exR,ivydv_bexI_p_exI,ivydv_bxx_m_bee_m_xx_p_ee, ivzdv_bexR_m_exR,ivzdv_bexI_p_exI,ivzdv_bxx_m_bee_m_xx_p_ee)
        #pragma omp simd
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            idv_bexR_m_exR            += vw[v] *       (in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
            idv_bexI_p_exI            += vw[v] *       (in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
            idv_bxx_m_bee_m_xx_p_ee   += vw[v] *       (in->wf[ff::bmm][ijkv]-in->wf[ff::bee][ijkv]+in->wf[ff::ee][ijkv]-in->wf[ff::mm][ijkv] );

            // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
            ivxdv_bexR_m_exR          += vw[v] * vx[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
            ivxdv_bexI_p_exI          += vw[v] * vx[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
            ivxdv_bxx_m_bee_m_xx_p_ee += vw[v] * vx[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::bee][ijkv]+in->wf[ff::ee][ijkv]-in->wf[ff::mm][ijkv] );
            ivydv_bexR_m_exR          += vw[v] * vy[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
            ivydv_bexI_p_exI          += vw[v] * vy[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
            ivydv_bxx_m_bee_m_xx_p_ee += vw[v] * vy[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::bee][ijkv]+in->wf[ff::ee][ijkv]-in->wf[ff::mm][ijkv] );
            ivzdv_bexR_m_exR          += vw[v] * vz[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
            ivzdv_bexI_p_exI          += vw[v] * vz[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
            ivzdv_bxx_m_bee_m_xx_p_ee += vw[v] * vz[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::bee][ijkv]+in->wf[ff::ee][ijkv]-in->wf[ff::mm][ijkv] );
        }
        //cout << "===" << nx << " " << nz << " " << nv << " " << gx << " " << gz << endl; 

        #pragma acc loop
        #pragma omp simd
        for (int v=0;v<nv; ++v) {

            auto ijkv = idx(i,j,k,v);

            // The base pointer for this stencil
            real *ee    = &(in->wf[ff::ee]  [ijkv]);
            real *xx    = &(in->wf[ff::mm]  [ijkv]);
            real *exr   = &(in->wf[ff::emr] [ijkv]);
            real *exi   = &(in->wf[ff::emi] [ijkv]);
            real *bee   = &(in->wf[ff::bee] [ijkv]);
            real *bxx   = &(in->wf[ff::bmm] [ijkv]);
            real *bexr  = &(in->wf[ff::bemr][ijkv]);
            real *bexi  = &(in->wf[ff::bemi][ijkv]);

            // prepare KO operator
#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps = -ko/dx/64.0;
#define KO_FD(x) ko_eps*( \
              ( x[-3*nv] + x[3*nv] - 6*(x[-2*nv] +x[2*nv])  + 15*(x[-nv] + x[nv]) - 20*x[0] ) + \
              ( x[-3*nzv]     + x[3*nzv]     - 6*(x[-2*nzv]     +x[2*nzv])      + 15*(x[-nzv]     + x[nzv])     - 20*x[0] ) + \
              ( x[-3*nyzv]    + x[3*nyzv]    - 6*(x[-2*nyzv]    +x[2*nyzv])     + 15*(x[-nyzv]    + x[nyzv])    - 20*x[0] ) )

#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps = -ko/dx/16.0;
#define KO_FD(x) ko_eps*( \
             ( x[-2*nv] + x[2*nv] - 4*(x[-nv] + x[nv]) + 6*x[0] ) + \
             ( x[-2*nzv]     + x[2*nzv]     - 4*(x[-nzv]     + x[nzv])     + 6*x[0] ) + \
             ( x[-2*nyzv]    + x[2*nyzv]    - 4*(x[-nyzv]    + x[nyzv])    + 6*x[0] ) )
#endif

            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor_z = -vz[v]/(12*dx);
            real factor_y = -vy[v]/(12*dx);
            real factor_x = -vx[v]/(12*dx);
#ifdef ADVEC_OFF
#define ADV_FD(x)     (0.0)
#else
#define ADV_FD(x) ( \
              factor_z*(  (x[-2*nv] - x[2*nv]) - 8.0*( x[-nv] -x[nv] ) ) + \
              factor_y*(  (x[-2*nzv]     - x[2*nzv])     - 8.0*( x[-nzv]     -x[nzv]     ) ) + \
              factor_x*(  (x[-2*nyzv]    - x[2*nyzv])    - 8.0*( x[-nyzv]    -x[nyzv]    ) ) )

#endif
            // interaction term
            real Iee    = 2*mu* (         exr[0]  *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bexR_m_exR          -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Iexr   =   mu* (   (xx[0]-ee[0]) *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Iexi   =   mu* (   (xx[0]-ee[0]) *(idv_bexR_m_exR -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) -  exr[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibee   = 2*mu* (        bexr[0]  *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bexR_m_exR          -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Ibexr  =   mu* ( (bxx[0]-bee[0]) *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibexi  =   mu* ( (bee[0]-bxx[0]) *(idv_bexR_m_exR -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) + bexr[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );

#ifndef VACUUM_OFF
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->wf[ff::ee]  [ijkv] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo* 2*st*exi [0];
            out->wf[ff::mm]  [ijkv] = -Iee   + ADV_FD(xx)   + KO_FD(xx)   + pmo* 2*st*exi [0];
            out->wf[ff::emr] [ijkv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr)  - pmo* 2*ct*exi [0];
            out->wf[ff::emi] [ijkv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi)  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->wf[ff::bee] [ijkv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo* 2*st*bexi[0];
            out->wf[ff::bmm] [ijkv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx)  + pmo* 2*st*bexi[0];
            out->wf[ff::bemr][ijkv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr) - pmo* 2*ct*bexi[0];
            out->wf[ff::bemi][ijkv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi) + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );
#else
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->wf[ff::ee]  [ijkv] =  Iee   + ADV_FD(ee)   + KO_FD(ee);
            out->wf[ff::mm]  [ijkv] = -Iee   + ADV_FD(xx)   + KO_FD(xx);
            out->wf[ff::emr] [ijkv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr);
            out->wf[ff::emi] [ijkv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi);
            out->wf[ff::bee] [ijkv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee);
            out->wf[ff::bmm] [ijkv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx);
            out->wf[ff::bemr][ijkv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr);
            out->wf[ff::bemi][ijkv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi);
#endif
        }
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush("VecMA");
#endif

    PARFORALL(i,j,k,v) {
        auto ijkv = idx(i,j,k,v);
        for(int f=0;f<nvar;++f)
          v0->wf[f][ijkv] = v1->wf[f][ijkv] + a * v2->wf[f][ijkv];
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2, const FieldVar * __restrict v3) {
#ifdef NVTX
    nvtxRangePush("Vec");
#endif

    PARFORALL(i,j,k,v) {
        auto ijkv = idx(i,j,k,v);
        for(int f=0;f<nvar;++f)
          v0->wf[f][ijkv] = v1->wf[f][ijkv] + a * (v2->wf[f][ijkv] + v3->wf[f][ijkv]);
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::step_rk4() {
#ifdef NVTX
    nvtxRangePush("Step");
#endif
    #ifdef PROFILING_BREAKDOWNS
    auto t0 = std::chrono::high_resolution_clock::now();
    #endif

    //Step-1
    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5*dt, v_rhs);
    //Step-2
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_pre, v_stat, 0.5*dt, v_cor);
    //Step-3
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_pre, v_stat, dt, v_cor);
    //Step-4
    calRHS(v_cor, v_pre);
    vectorize(v_stat, v_stat, 1.0/6.0*dt, v_cor, v_rhs);

    #ifdef PROFILING_BREAKDOWNS
    t_step += std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() -t0 ).count();
    #endif
    if(renorm) renormalize(v_stat);
    phy_time += dt;
#ifdef NVTX
    nvtxRangePop();
#endif
}
