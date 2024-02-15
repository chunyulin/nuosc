#include "nuosc_class.h"

void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("calRHS");
#endif

    // for abbreviation
    const int nzv  = nv* (nx[2] + 2*gx[2]);
    const int nyzv = nzv*(nx[1] + 2*gx[1]);
// gang worker num_workers(64)
#pragma omp parallel for collapse(3)
#pragma acc parallel loop independent collapse(3)
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<nx[2]; ++k) {

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
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            idv_bexR_m_exR            += vw[v] *       (in->bex_re[ijkv] - in->ex_re[ijkv] );
            idv_bexI_p_exI            += vw[v] *       (in->bex_im[ijkv] + in->ex_im[ijkv] );
            idv_bxx_m_bee_m_xx_p_ee   += vw[v] *       (in->bxx[ijkv]-in->bee[ijkv]+in->ee[ijkv]-in->xx[ijkv] );

            // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
            ivxdv_bexR_m_exR          += vw[v] * vx[v]*(in->bex_re[ijkv] - in->ex_re[ijkv] );
            ivxdv_bexI_p_exI          += vw[v] * vx[v]*(in->bex_im[ijkv] + in->ex_im[ijkv] );
            ivxdv_bxx_m_bee_m_xx_p_ee += vw[v] * vx[v]*(in->bxx[ijkv]-in->bee[ijkv]+in->ee[ijkv]-in->xx[ijkv] );
            ivydv_bexR_m_exR          += vw[v] * vy[v]*(in->bex_re[ijkv] - in->ex_re[ijkv] );
            ivydv_bexI_p_exI          += vw[v] * vy[v]*(in->bex_im[ijkv] + in->ex_im[ijkv] );
            ivydv_bxx_m_bee_m_xx_p_ee += vw[v] * vy[v]*(in->bxx[ijkv]-in->bee[ijkv]+in->ee[ijkv]-in->xx[ijkv] );
            ivzdv_bexR_m_exR          += vw[v] * vz[v]*(in->bex_re[ijkv] - in->ex_re[ijkv] );
            ivzdv_bexI_p_exI          += vw[v] * vz[v]*(in->bex_im[ijkv] + in->ex_im[ijkv] );
            ivzdv_bxx_m_bee_m_xx_p_ee += vw[v] * vz[v]*(in->bxx[ijkv]-in->bee[ijkv]+in->ee[ijkv]-in->xx[ijkv] );
        }
        //cout << "===" << nx << " " << nz << " " << nv << " " << gx << " " << gz << endl; 

        #pragma acc loop
        for (int v=0;v<nv; ++v) {

            auto ijkv = idx(i,j,k,v);

            // The base pointer for this stencil
            real *ee    = &(in->ee    [ijkv]);
            real *xx    = &(in->xx    [ijkv]);
            real *exr   = &(in->ex_re [ijkv]);
            real *exi   = &(in->ex_im [ijkv]);
            real *bee   = &(in->bee   [ijkv]);
            real *bxx   = &(in->bxx   [ijkv]);
            real *bexr  = &(in->bex_re[ijkv]);
            real *bexi  = &(in->bex_im[ijkv]);

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
            out->ee    [ijkv] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo* 2*st*exi [0];
            out->xx    [ijkv] = -Iee   + ADV_FD(xx)   + KO_FD(xx)   + pmo* 2*st*exi [0];
            out->ex_re [ijkv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr)  - pmo* 2*ct*exi [0];
            out->ex_im [ijkv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi)  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [ijkv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo* 2*st*bexi[0];
            out->bxx   [ijkv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx)  + pmo* 2*st*bexi[0];
            out->bex_re[ijkv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr) - pmo* 2*ct*bexi[0];
            out->bex_im[ijkv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi) + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );
#else
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [ijkv] =  Iee   + ADV_FD(ee)   + KO_FD(ee);
            out->xx    [ijkv] = -Iee   + ADV_FD(xx)   + KO_FD(xx);
            out->ex_re [ijkv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr);
            out->ex_im [ijkv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi);
            out->bee   [ijkv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee);
            out->bxx   [ijkv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx);
            out->bex_re[ijkv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr);
            out->bex_im[ijkv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi);
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
        v0->ee    [ijkv] = v1->ee    [ijkv] + a * v2->ee    [ijkv];
        v0->xx    [ijkv] = v1->xx    [ijkv] + a * v2->xx    [ijkv];
        v0->ex_re [ijkv] = v1->ex_re [ijkv] + a * v2->ex_re [ijkv];
        v0->ex_im [ijkv] = v1->ex_im [ijkv] + a * v2->ex_im [ijkv];
        v0->bee   [ijkv] = v1->bee   [ijkv] + a * v2->bee   [ijkv];
        v0->bxx   [ijkv] = v1->bxx   [ijkv] + a * v2->bxx   [ijkv];
        v0->bex_re[ijkv] = v1->bex_re[ijkv] + a * v2->bex_re[ijkv];
        v0->bex_im[ijkv] = v1->bex_im[ijkv] + a * v2->bex_im[ijkv];
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
        v0->ee    [ijkv] = v1->ee    [ijkv] + a * (v2->ee    [ijkv] + v3->ee    [ijkv]);
        v0->xx    [ijkv] = v1->xx    [ijkv] + a * (v2->xx    [ijkv] + v3->xx    [ijkv]);
        v0->ex_re [ijkv] = v1->ex_re [ijkv] + a * (v2->ex_re [ijkv] + v3->ex_re [ijkv]);
        v0->ex_im [ijkv] = v1->ex_im [ijkv] + a * (v2->ex_im [ijkv] + v3->ex_im [ijkv]);
        v0->bee   [ijkv] = v1->bee   [ijkv] + a * (v2->bee   [ijkv] + v3->bee   [ijkv]);
        v0->bxx   [ijkv] = v1->bxx   [ijkv] + a * (v2->bxx   [ijkv] + v3->bxx   [ijkv]);
        v0->bex_re[ijkv] = v1->bex_re[ijkv] + a * (v2->bex_re[ijkv] + v3->bex_re[ijkv]);
        v0->bex_im[ijkv] = v1->bex_im[ijkv] + a * (v2->bex_im[ijkv] + v3->bex_im[ijkv]);
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::step_rk4() {
#ifdef NVTX
    nvtxRangePush("Step");
#endif

    #ifdef PROFILING
    auto t0 = std::chrono::high_resolution_clock::now();
    #endif

    //Step-1
#ifdef COSENU_MPI
    sync_boundary(v_stat);
#else
    updatePeriodicBoundary(v_stat);
#endif

    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5*dt, v_rhs);

    //Step-2
#ifdef COSENU_MPI
    sync_boundary(v_pre);
#else
    updatePeriodicBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, 0.5*dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-3
#ifdef COSENU_MPI
    sync_boundary(v_pre);
#else
    updatePeriodicBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-4
#ifdef COSENU_MPI
    sync_boundary(v_pre);
#else
    updatePeriodicBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_pre, v_stat, 1.0/6.0*dt, v_cor, v_rhs);
    swap(&v_pre, &v_stat);

    #ifdef PROFILING
    t_step += std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() -t0 ).count();
    #endif

    if(renorm) renormalize(v_stat);

    phy_time += dt;
#ifdef NVTX
    nvtxRangePop();
#endif
}
