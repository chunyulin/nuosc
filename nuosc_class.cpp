#include "nuosc_class.h"

void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("calRHS");
#endif

    // for abbreviation
    auto vw = grid.vw.data();
    auto vx = grid.vx.data();
    auto vy = grid.vy.data();
    auto vz = grid.vz.data();
    const int nzv  = grid.nv*(grid.nx[2]+2*grid.gx[2]);
    const int nyzv = nzv*(grid.nx[1] + 2*grid.gx[1]);

#pragma omp parallel for collapse(3)
#pragma acc parallel loop independent collapse(3)
    for (int i=0;i<grid.nx[0]; ++i)
    for (int j=0;j<grid.nx[1]; ++j)
    for (int k=0;k<grid.nx[2]; ++k) {

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
        for (int v=0;v<grid.nv; ++v) {
            auto ijkv = grid.idx(i,j,k,v);
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
        //cout << "===" << grid.nx << " " << grid.nz << " " << grid.nv << " " << grid.gx << " " << grid.gz << endl; 

        #pragma acc loop
        for (int v=0;v<grid.nv; ++v) {

            auto ijkv = grid.idx(i,j,k,v);

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
            real ko_eps = -ko/grid.dx/64.0;
#define KO_FD(x) ko_eps*( \
              ( x[-3*grid.nv] + x[3*grid.nv] - 6*(x[-2*grid.nv] +x[2*grid.nv])  + 15*(x[-grid.nv] + x[grid.nv]) - 20*x[0] ) + \
              ( x[-3*nzv]     + x[3*nzv]     - 6*(x[-2*nzv]     +x[2*nzv])      + 15*(x[-nzv]     + x[nzv])     - 20*x[0] ) + \
              ( x[-3*nyzv]    + x[3*nyzv]    - 6*(x[-2*nyzv]    +x[2*nyzv])     + 15*(x[-nyzv]    + x[nyzv])    - 20*x[0] ) )

#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps = -ko/grid.dx/16.0;
#define KO_FD(x) ko_eps*( \
             ( x[-2*grid.nv] + x[2*grid.nv] - 4*(x[-grid.nv] + x[grid.nv]) + 6*x[0] ) + \
             ( x[-2*nzv]     + x[2*nzv]     - 4*(x[-nzv]     + x[nzv])     + 6*x[0] ) + \
             ( x[-2*nyzv]    + x[2*nyzv]    - 4*(x[-nyzv]    + x[nyzv])    + 6*x[0] ) )
#endif

            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor_z = -grid.vz[v]/(12*grid.dx);
            real factor_y = -grid.vy[v]/(12*grid.dx);
            real factor_x = -grid.vx[v]/(12*grid.dx);
#ifdef ADVEC_OFF
#define ADV_FD(x)     (0.0)
#else
#define ADV_FD(x) ( \
              factor_z*(  (x[-2*grid.nv] - x[2*grid.nv]) - 8.0*( x[-grid.nv] -x[grid.nv] ) ) + \
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
    nvtxRangePush("vectorize");
#endif

    PARFORALL(i,j,k,v) {
        auto ijkv = grid.idx(i,j,k,v);
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
    nvtxRangePush("vectorize");
#endif

    PARFORALL(i,j,k,v) {
        auto ijkv = grid.idx(i,j,k,v);
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
    nvtxRangePush("step_rk4");
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

    if(renorm) renormalize(v_stat);

    phy_time += dt;
#ifdef NVTX
    nvtxRangePop();
#endif
}

NuOsc::NuOsc(int px_[], int nv_, const int nphi_, const int gx_[],
             const real bbox_[][2], const real dx_, const real CFL_, const real  ko_) :
                 phy_time(0.), ko(ko_), dx(dx_), grid(px_,nv_,nphi_,gx_,bbox_, dx_) {

        auto size = grid.get_lpts();
        real mem_per_var = size*8/1024./1024./1024;

        ds_L = dx*dx*dx/(bbox_[0][1]-bbox_[0][0])/(bbox_[1][1]-bbox_[1][0])/(bbox_[2][1]-bbox_[2][0]);

        CFL = CFL_;
        dt = dx*CFL;

#ifdef COSENU_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

#endif
 
    //MPI_Barrier(MPI_COMM_WORLD);

    if (myrank==0) {
            printf("\nNuOsc on %d MPI ranks: %d core per rank.\n", grid.ranks, omp_get_max_threads() );
            printf("   Domain:  v: nv = %5d  ( w/ nphi = %5d ) on S2.\n", grid.get_nv(), grid.get_nphi() );
            printf("            x:( %12f %12f )  dx = %g\n", bbox_[0][0], bbox_[0][1], dx);
            printf("            y:( %12f %12f )  dy = %g\n", bbox_[1][0], bbox_[1][1], dx);
            printf("            z:( %12f %12f )  dz = %g\n", bbox_[2][0], bbox_[2][1], dx);
            printf("   Local size per field var = %.2f GB, totol memory per rank roughly %.2f GB\n", mem_per_var, mem_per_var*50);
            printf("   dt = %g     CFL = %g\n", dt, CFL);
#ifdef BC_PERI
            printf("   Use Periodic boundary\n");
#else
            printf("   Use open boundary\n");
#endif

#if defined(IM_V2D_POLAR_GL_Z)
            printf("   Use Gauss-Lobatto z-grid and uniform phi-grid.\n");
#else
            printf("   Use uniform z- and phi- grid.\n");
#endif

#ifndef KO_ORD_3
            printf("   Use 5-th order KO dissipation, KO eps = %g\n", ko);
#else
            printf("   Use 3-th order KO dissipation, KO eps = %g\n", ko);
#endif

#ifndef ADVEC_OFF
            printf("   Advection ON. (Center-FD)\n");
            //printf("   Use upwinded for advaction. (EXP. Always blowup!!\n");
            //printf("   Use lopsided FD for advaction\n");
#else
            printf("   Advection OFF.\n");
#endif

#ifdef VACUUM_OFF
            printf("   Vacuum term OFF.\n");
#else
            printf("   Vacuum term ON:  pmo= %g  theta= %g.\n", pmo, theta);
#endif
        }

        // field variables for analysis~~
        G0  = new real[size];
        G0b = new real[size];
        P1  = new real[size];
        P2  = new real[size];
        P3  = new real[size];
        P1b = new real[size];
        P2b = new real[size];
        P3b = new real[size];
        dP  = new real[size];
        dN  = new real[size];
        dPb = new real[size];
        dNb = new real[size];
#pragma acc enter data create(G0[0:size],G0b[0:size],P1[0:size],P2[0:size],P3[0:size],P1b[0:size],P2b[0:size],P3b[0:size],dP[0:size],dN[0:size],dPb[0:size],dNb[0:size])

        // field variables~~
        v_stat = new FieldVar(size);
        v_rhs  = new FieldVar(size);
        v_pre  = new FieldVar(size);
        v_cor  = new FieldVar(size);
        v_stat0 = new FieldVar(size);
#pragma acc enter data create(v_stat[0:1], v_stat0[0:1], v_rhs[0:1], v_pre[0:1], v_cor[0:1]) attach(v_stat, v_rhs, v_pre, v_cor, v_stat0)

        if (myrank==0) {
            anafile.open("analysis.dat", std::ofstream::out | std::ofstream::trunc);
            if(!anafile) cout << "*** Open fails: " << "./analysis.dat" << endl;
            anafile << "### [ phy_time,   1:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0    7:Lex   8:ELNe]" << endl;
        }

    }

