#include "nuosc_class.h"


void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#ifdef COSENU2D
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3)
    for (int i=0;i<ny; i++)
#else
    int i=0;
    #pragma omp parallel for collapse(2)
    #pragma acc parallel loop collapse(2)
#endif
    for (int j=0;j<gz; j++)
    for (int v=0;v<nv; v++) {
                //z lower side
                in->ee    [idx(i,-j-1,v)] = in->ee    [idx(i,nz-j-1,v)];
                in->xx    [idx(i,-j-1,v)] = in->xx    [idx(i,nz-j-1,v)];
                in->ex_re [idx(i,-j-1,v)] = in->ex_re [idx(i,nz-j-1,v)];
                in->ex_im [idx(i,-j-1,v)] = in->ex_im [idx(i,nz-j-1,v)];
                in->bee   [idx(i,-j-1,v)] = in->bee   [idx(i,nz-j-1,v)];
                in->bxx   [idx(i,-j-1,v)] = in->bxx   [idx(i,nz-j-1,v)];
                in->bex_re[idx(i,-j-1,v)] = in->bex_re[idx(i,nz-j-1,v)];
                in->bex_im[idx(i,-j-1,v)] = in->bex_im[idx(i,nz-j-1,v)];
                //z upper side
                in->ee    [idx(i,nz+j,v)] = in->ee    [idx(i,j,v)];
                in->xx    [idx(i,nz+j,v)] = in->xx    [idx(i,j,v)];
                in->ex_re [idx(i,nz+j,v)] = in->ex_re [idx(i,j,v)];
                in->ex_im [idx(i,nz+j,v)] = in->ex_im [idx(i,j,v)];
                in->bee   [idx(i,nz+j,v)] = in->bee   [idx(i,j,v)];
                in->bxx   [idx(i,nz+j,v)] = in->bxx   [idx(i,j,v)];
                in->bex_re[idx(i,nz+j,v)] = in->bex_re[idx(i,j,v)];
                in->bex_im[idx(i,nz+j,v)] = in->bex_im[idx(i,j,v)];
    }

#ifdef COSENU2D
#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
    for (int i=0;i<gy; i++)
    for (int j=0;j<nz; j++)
    for (int v=0;v<nv; v++) {
                //y lower side
                in->ee    [idx(-i-1,j,v)] = in->ee    [idx(ny-i-1,j,v)];
                in->xx    [idx(-i-1,j,v)] = in->xx    [idx(ny-i-1,j,v)];
                in->ex_re [idx(-i-1,j,v)] = in->ex_re [idx(ny-i-1,j,v)];
                in->ex_im [idx(-i-1,j,v)] = in->ex_im [idx(ny-i-1,j,v)];
                in->bee   [idx(-i-1,j,v)] = in->bee   [idx(ny-i-1,j,v)];
                in->bxx   [idx(-i-1,j,v)] = in->bxx   [idx(ny-i-1,j,v)];
                in->bex_re[idx(-i-1,j,v)] = in->bex_re[idx(ny-i-1,j,v)];
                in->bex_im[idx(-i-1,j,v)] = in->bex_im[idx(ny-i-1,j,v)];
                //y upper side
                in->ee    [idx(ny+i,j,v)] = in->ee    [idx(i,j,v)];
                in->xx    [idx(ny+i,j,v)] = in->xx    [idx(i,j,v)];
                in->ex_re [idx(ny+i,j,v)] = in->ex_re [idx(i,j,v)];
                in->ex_im [idx(ny+i,j,v)] = in->ex_im [idx(i,j,v)];
                in->bee   [idx(ny+i,j,v)] = in->bee   [idx(i,j,v)];
                in->bxx   [idx(ny+i,j,v)] = in->bxx   [idx(i,j,v)];
                in->bex_re[idx(ny+i,j,v)] = in->bex_re[idx(i,j,v)];
                in->bex_im[idx(ny+i,j,v)] = in->bex_im[idx(i,j,v)];
            }
#endif

#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { 
    cout << "Not implemented." << endl;
    assert(0);
}

void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    const int nzv = nv*nz;

#ifdef COSENU2D
    #pragma omp parallel for collapse(2)
    #pragma acc parallel loop independent collapse(2)
    for (int i=0;i<ny; i++)
#else
    int i = 0;
    #pragma omp parallel for
    #pragma acc parallel loop num_gangs(8192)
#endif
    for (int j=0;j<nz; j++) {

        // common integral over vz'
        real idv_bexR_m_exR  = 0;
        real idv_bexI_p_exI  = 0;
        real ivdv_bexR_m_exR = 0;
        real ivdv_bexI_p_exI = 0;
        real idv_bxx_m_bee_m_xx_p_ee  = 0;
        real ivdv_bxx_m_bee_m_xx_p_ee = 0;

        #pragma acc loop reduction(+:idv_bexR_m_exR,idv_bexI_p_exI,idv_bxx_m_bee_m_xx_p_ee,ivdv_bexR_m_exR,ivdv_bexI_p_exI,ivdv_bxx_m_bee_m_xx_p_ee)
        for (int k=0;k<nv; k++) {
             idv_bexR_m_exR  += vw[k] *       (in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             idv_bexI_p_exI  += vw[k] *       (in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             ivdv_bexR_m_exR += vw[k] * vz[k]*(in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             ivdv_bexI_p_exI += vw[k] * vz[k]*(in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             idv_bxx_m_bee_m_xx_p_ee  += vw[k] *       (in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
             ivdv_bxx_m_bee_m_xx_p_ee += vw[k] * vz[k]*(in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
        }

        #pragma acc loop
        for (int v=0;v<nv; v++) {

            uint ijv = idx(i,j,v);

            // The base pointer for this stencil
            real *ee    = &(in->ee    [ijv]);
            real *xx    = &(in->xx    [ijv]);
            real *exr   = &(in->ex_re [ijv]);
            real *exi   = &(in->ex_im [ijv]);
            real *bee   = &(in->bee   [ijv]);
            real *bxx   = &(in->bxx   [ijv]);
            real *bexr  = &(in->bex_re[ijv]);
            real *bexi  = &(in->bex_im[ijv]);

            // prepare KO operator
#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps_z = -ko/dz/64.0;
  #ifdef COSENU2D
            real ko_eps_y = -ko/dy/64.0;
            #define KO_FD(x) ( ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] ) + \
                               ko_eps_y*( x[-3*nzv] + x[3*nzv] - 6*(x[-2*nzv]+x[2*nzv]) + 15*(x[-nzv]+x[nzv]) - 20*x[0] ) )
  #else
            #define KO_FD(x)   ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] )
  #endif
#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps_z = -ko/dz/16.0;
  #ifdef COSENU2D
            real ko_eps_y = -ko/dy/16.0;
            #define KO_FD(x) ( ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] ) + \
                               ko_eps_y * ( x[-2*nzv] + x[2*nzv] - 4*(x[-nzv]+x[nzv]) + 6*x[0] ) )
  #else
	    #define KO_FD(x)   ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] )
  #endif
#endif

            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor_z = -vz[v]/(12*dz);
#ifdef ADVEC_OFF
            #define ADV_FD(x)     (0.0)
#else
  #ifdef COSENU2D
            real factor_y = -vy[v]/(12*dy);
            #define ADV_FD(x) ( factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv]  ) ) + \
                                  factor_y*(  (x[-2*nzv]-x[2*nzv]) - 8.0*(x[-nzv]-x[nzv] ) ) )
  #else
            #define ADV_FD(x)   factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv])  )
  #endif

#endif

            // interaction term
            real Iee    = 2*mu* (         exr[0]  *(idv_bexI_p_exI - vz[v]*ivdv_bexI_p_exI ) +  exi[0]*(idv_bexR_m_exR          - vz[v]*ivdv_bexR_m_exR )  );
            real Iexr   =   mu* (   (xx[0]-ee[0]) *(idv_bexI_p_exI - vz[v]*ivdv_bexI_p_exI ) +  exi[0]*(idv_bxx_m_bee_m_xx_p_ee - vz[v]*ivdv_bxx_m_bee_m_xx_p_ee) );
            real Iexi   =   mu* (   (xx[0]-ee[0]) *(idv_bexR_m_exR - vz[v]*ivdv_bexR_m_exR ) -  exr[0]*(idv_bxx_m_bee_m_xx_p_ee - vz[v]*ivdv_bxx_m_bee_m_xx_p_ee) );
            real Ibee   = 2*mu* (        bexr[0]  *(idv_bexI_p_exI - vz[v]*ivdv_bexI_p_exI ) - bexi[0]*(idv_bexR_m_exR          - vz[v]*ivdv_bexR_m_exR )  );
            real Ibexr  =   mu* ( (bxx[0]-bee[0]) *(idv_bexI_p_exI - vz[v]*ivdv_bexI_p_exI ) - bexi[0]*(idv_bxx_m_bee_m_xx_p_ee - vz[v]*ivdv_bxx_m_bee_m_xx_p_ee) );
            real Ibexi  =   mu* ( (bee[0]-bxx[0]) *(idv_bexR_m_exR - vz[v]*ivdv_bexR_m_exR ) + bexr[0]*(idv_bxx_m_bee_m_xx_p_ee - vz[v]*ivdv_bxx_m_bee_m_xx_p_ee) );

#ifndef VACUUM_OFF
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [ijv] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo* 2*st*exi [0];
            out->xx    [ijv] = -Iee   + ADV_FD(xx)   + KO_FD(xx)   + pmo* 2*st*exi [0];
            out->ex_re [ijv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr)  - pmo* 2*ct*exi [0];
            out->ex_im [ijv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi)  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [ijv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo* 2*st*bexi[0];
            out->bxx   [ijv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx)  + pmo* 2*st*bexi[0];
            out->bex_re[ijv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr) - pmo* 2*ct*bexi[0];
            out->bex_im[ijv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi) + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );
#else
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [ijv] =  Iee   + ADV_FD(ee)   + KO_FD(ee);
            out->xx    [ijv] = -Iee   + ADV_FD(xx)   + KO_FD(xx);
            out->ex_re [ijv] =  Iexr  + ADV_FD(exr)  + KO_FD(exr);
            out->ex_im [ijv] =  Iexi  + ADV_FD(exi)  + KO_FD(exi);
            out->bee   [ijv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee);
            out->bxx   [ijv] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx);
            out->bex_re[ijv] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr);
            out->bex_im[ijv] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi);
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
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
        auto k = idx(i,j,v);
        v0->ee    [k] = v1->ee    [k] + a * v2->ee    [k];
        v0->xx    [k] = v1->xx    [k] + a * v2->xx    [k];
        v0->ex_re [k] = v1->ex_re [k] + a * v2->ex_re [k];
        v0->ex_im [k] = v1->ex_im [k] + a * v2->ex_im [k];
        v0->bee   [k] = v1->bee   [k] + a * v2->bee   [k];
        v0->bxx   [k] = v1->bxx   [k] + a * v2->bxx   [k];
        v0->bex_re[k] = v1->bex_re[k] + a * v2->bex_re[k];
        v0->bex_im[k] = v1->bex_im[k] + a * v2->bex_im[k];
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2, const FieldVar * __restrict v3) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
            auto k = idx(i,j,v);
            v0->ee    [k] = v1->ee    [k] + a * (v2->ee    [k] + v3->ee    [k]);
            v0->xx    [k] = v1->xx    [k] + a * (v2->xx    [k] + v3->xx    [k]);
            v0->ex_re [k] = v1->ex_re [k] + a * (v2->ex_re [k] + v3->ex_re [k]);
            v0->ex_im [k] = v1->ex_im [k] + a * (v2->ex_im [k] + v3->ex_im [k]);
            v0->bee   [k] = v1->bee   [k] + a * (v2->bee   [k] + v3->bee   [k]);
            v0->bxx   [k] = v1->bxx   [k] + a * (v2->bxx   [k] + v3->bxx   [k]);
            v0->bex_re[k] = v1->bex_re[k] + a * (v2->bex_re[k] + v3->bex_re[k]);
            v0->bex_im[k] = v1->bex_im[k] + a * (v2->bex_im[k] + v3->bex_im[k]);
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::step_rk4() {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    //Step-1
#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif
    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5*dt, v_rhs);

    //Step-2
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, 0.5*dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-3
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-4
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
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



