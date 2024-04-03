#include "nuosc_class.h"

void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("PeriodicBoundary");
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#ifdef COSENU2D
    #pragma omp parallel for collapse(3)
    #pragma acc parallel loop collapse(3)
    for (int i=0;i<nx; ++i)
#else
    int i=0;
    #pragma omp parallel for simd collapse(2)
    #pragma acc parallel loop collapse(2)
#endif
    for (int j=0;j<gz; ++j)
    for (int v=0;v<nv; ++v) {
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
    for (int i=0;i<gx; ++i)
    for (int j=0;j<nz; ++j)
    for (int v=0;v<nv; ++v) {
                //y lower side
                in->ee    [idx(-i-1,j,v)] = in->ee    [idx(nx-i-1,j,v)];
                in->xx    [idx(-i-1,j,v)] = in->xx    [idx(nx-i-1,j,v)];
                in->ex_re [idx(-i-1,j,v)] = in->ex_re [idx(nx-i-1,j,v)];
                in->ex_im [idx(-i-1,j,v)] = in->ex_im [idx(nx-i-1,j,v)];
                in->bee   [idx(-i-1,j,v)] = in->bee   [idx(nx-i-1,j,v)];
                in->bxx   [idx(-i-1,j,v)] = in->bxx   [idx(nx-i-1,j,v)];
                in->bex_re[idx(-i-1,j,v)] = in->bex_re[idx(nx-i-1,j,v)];
                in->bex_im[idx(-i-1,j,v)] = in->bex_im[idx(nx-i-1,j,v)];
                //y upper side
                in->ee    [idx(nx+i,j,v)] = in->ee    [idx(i,j,v)];
                in->xx    [idx(nx+i,j,v)] = in->xx    [idx(i,j,v)];
                in->ex_re [idx(nx+i,j,v)] = in->ex_re [idx(i,j,v)];
                in->ex_im [idx(nx+i,j,v)] = in->ex_im [idx(i,j,v)];
                in->bee   [idx(nx+i,j,v)] = in->bee   [idx(i,j,v)];
                in->bxx   [idx(nx+i,j,v)] = in->bxx   [idx(i,j,v)];
                in->bex_re[idx(nx+i,j,v)] = in->bex_re[idx(i,j,v)];
                in->bex_im[idx(nx+i,j,v)] = in->bex_im[idx(i,j,v)];
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
    nvtxRangePush("calRHS");
#endif

    #define nzv nv*(nz+2*gz)

#ifdef COSENU2D
    #pragma omp parallel for collapse(2)
    #pragma acc parallel loop independent collapse(2)
    for (int i=0;i<nx; ++i)
#else
    int i = 0;
    #pragma omp parallel for
    #pragma acc parallel loop independent num_gangs(8192)
#endif
    for (int j=0;j<nz; ++j) {

        // common integral over vz'
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

        // OMP reduction not useful here
#ifdef COSENU2D
        #pragma acc loop reduction(+:idv_bexR_m_exR,idv_bexI_p_exI,idv_bxx_m_bee_m_xx_p_ee, ivxdv_bexR_m_exR,ivxdv_bexI_p_exI,ivxdv_bxx_m_bee_m_xx_p_ee, ivydv_bexR_m_exR,ivydv_bexI_p_exI,ivydv_bxx_m_bee_m_xx_p_ee, ivzdv_bexR_m_exR,ivzdv_bexI_p_exI,ivzdv_bxx_m_bee_m_xx_p_ee)
#else
        #pragma acc loop reduction(+:idv_bexR_m_exR,idv_bexI_p_exI,idv_bxx_m_bee_m_xx_p_ee,ivzdv_bexR_m_exR,ivzdv_bexI_p_exI,ivzdv_bxx_m_bee_m_xx_p_ee)
        #pragma omp simd
#endif
        for (int k=0;k<nv; ++k) {
             idv_bexR_m_exR            += vw[k] *       (in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             idv_bexI_p_exI            += vw[k] *       (in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             idv_bxx_m_bee_m_xx_p_ee   += vw[k] *       (in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
#ifdef COSENU2D
             // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
             ivxdv_bexR_m_exR          += vw[k] * vx[k]*(in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             ivxdv_bexI_p_exI          += vw[k] * vx[k]*(in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             ivxdv_bxx_m_bee_m_xx_p_ee += vw[k] * vx[k]*(in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
             ivydv_bexR_m_exR          += vw[k] * vy[k]*(in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             ivydv_bexI_p_exI          += vw[k] * vy[k]*(in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             ivydv_bxx_m_bee_m_xx_p_ee += vw[k] * vy[k]*(in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
#endif
             ivzdv_bexR_m_exR          += vw[k] * vz[k]*(in->bex_re[idx(i,j,k)] - in->ex_re[idx(i,j,k)] );
             ivzdv_bexI_p_exI          += vw[k] * vz[k]*(in->bex_im[idx(i,j,k)] + in->ex_im[idx(i,j,k)] );
             ivzdv_bxx_m_bee_m_xx_p_ee += vw[k] * vz[k]*(in->bxx[idx(i,j,k)]-in->bee[idx(i,j,k)]+in->ee[idx(i,j,k)]-in->xx[idx(i,j,k)] );
       }

        #pragma acc loop
        #pragma omp simd
        for (int v=0;v<nv; ++v) {

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

#ifdef COSENU2D
            // interaction term
            real Iee    = 2*mu* (         exr[0]  *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bexR_m_exR          -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Iexr   =   mu* (   (xx[0]-ee[0]) *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Iexi   =   mu* (   (xx[0]-ee[0]) *(idv_bexR_m_exR -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) -  exr[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibee   = 2*mu* (        bexr[0]  *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bexR_m_exR          -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Ibexr  =   mu* ( (bxx[0]-bee[0]) *(idv_bexI_p_exI -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibexi  =   mu* ( (bee[0]-bxx[0]) *(idv_bexR_m_exR -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) + bexr[0]*(idv_bxx_m_bee_m_xx_p_ee -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
#else
            real Iee    = 2*mu* (         exr[0]  *(idv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Iexr   =   mu* (   (xx[0]-ee[0]) *(idv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) +  exi[0]*(idv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Iexi   =   mu* (   (xx[0]-ee[0]) *(idv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) -  exr[0]*(idv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibee   = 2*mu* (        bexr[0]  *(idv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bexR_m_exR          -vz[v]*ivzdv_bexR_m_exR         ) );
            real Ibexr  =   mu* ( (bxx[0]-bee[0]) *(idv_bexI_p_exI -vz[v]*ivzdv_bexI_p_exI ) - bexi[0]*(idv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
            real Ibexi  =   mu* ( (bee[0]-bxx[0]) *(idv_bexR_m_exR -vz[v]*ivzdv_bexR_m_exR ) + bexr[0]*(idv_bxx_m_bee_m_xx_p_ee -vz[v]*ivzdv_bxx_m_bee_m_xx_p_ee) );
#endif

            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [ijv] =  Iee   - pmo* 2*st*exi [0];
            out->xx    [ijv] = -Iee   + pmo* 2*st*exi [0];
            out->ex_re [ijv] =  Iexr  - pmo* 2*ct*exi [0];
            out->ex_im [ijv] =  Iexi  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [ijv] =  Ibee  - pmo* 2*st*bexi[0];
            out->bxx   [ijv] = -Ibee  + pmo* 2*st*bexi[0];
            out->bex_re[ijv] =  Ibexr - pmo* 2*ct*bexi[0];
            out->bex_im[ijv] =  Ibexi + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );
        } // end for v
    }  // end for z

    std::vector<real*> iffs = {in->ee, in->xx, in->ex_re, in->ex_im,
                           in->bee, in->bxx, in->bex_re, in->bex_im };
    std::vector<real*> offs = {out->ee, out->xx, out->ex_re, out->ex_im,
                           out->bee, out->bxx, out->bex_re, out->bex_im };
#ifdef WENO7
    for(int f; f<iffs.size(); ++f) {
        get_flux(flux, iffs[f], nv);
        PARFORALL(i,j,v) {
            auto ijkv = idx(0,j,v);
            int s = sgn(vz[v]);
            offs[f][ijkv] += -0.5*vz[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nv]) + std::abs(1-s)*(flux->h2l[ijkv+nv]-flux->h2l[ijkv]));
        }
    }
#else
    for(int f; f<iffs.size(); ++f) {
        PARFORALL(i,j,v) {

            // prepare KO operator
#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps_z = -ko/dz/64.0;
  #ifdef COSENU2D
            real ko_eps_y = -ko/dx/64.0;
            #define KO_FD(x) ( ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv] +x[2*nv])  + 15*(x[-nv] +x[nv])  - 20*x[0] ) + \
                               ko_eps_y*( x[-3*nzv] + x[3*nzv] - 6*(x[-2*nzv]+x[2*nzv]) + 15*(x[-nzv]+x[nzv]) - 20*x[0] ) )
  #else
            #define KO_FD(x)   ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] )
  #endif
#else
            // Kreiss-Oliger dissipation (3-nd order)
  #ifdef COSENU2D
            real ko_eps_x = -ko/dx/16.0;
            #define KO_FD(x) ( ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] ) + \
                               ko_eps_x * ( x[-2*nzv] + x[2*nzv] - 4*(x[-nzv]+x[nzv]) + 6*x[0] ) )
  #else
    #ifdef FD8
            // 3-th order with 4 buffer zone
            real ko_eps_z = -ko/dz/16.0 / 240.0;
            #define KO_FD(x) ( ko_eps_z*( 7.0*(x[-4*nv]+x[4*nv])-96.0*(x[-3*nv]+x[3*nv])+ \
                                        676.0*(x[-2*nv]+x[2*nv])-1952.0*(x[-nv]+x[nv])+2730.0*x[0] ) )
    #else
            real ko_eps_z = -ko/dz/16.0;
            #define KO_FD(x)   ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] )
    #endif
  #endif
#endif

            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
#ifdef ADVEC_OFF
            #define ADV_FD(x)     (0.0)
#else
  #ifdef COSENU2D
            real factor_z = -vz[v]/(12*dz);
            real factor_x = -vx[v]/(12*dx);
            #define ADV_FD(x) ( factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv]  ) ) + \
                                factor_x*(  (x[-2*nzv]-x[2*nzv]) - 8.0*(x[-nzv]-x[nzv] ) ) )
  #else
    #ifdef FD8
            real factor_z = -vz[v]/(280*dz);
            #define ADV_FD(x) factor_z*( (x[-4*nv]-x[4*nv]) - (224./21.0)*(x[-3*nv]-x[3*nv]) + 56.0*(x[-2*nv]-x[2*nv]) - 224.0*(x[-nv]-x[nv]))
    #else
            real factor_z = -vz[v]/(12*dz);
            #define ADV_FD(x)   factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv])  )
    #endif
  #endif

#endif    // end if ADVEC_OFF


            auto ijkv = idx(0,j,v);
            real *iff = &(iffs[f][ijkv]);
            int s = sgn(vz[v]);
            offs[f][ijkv] += ADV_FD(iff) + KO_FD(iff) ;
        }
    }



#endif   // end if WENO7



#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush("vectorize");
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
    nvtxRangePush("vectorize");
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
    nvtxRangePush("step_rk4");
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

#ifdef WENO7
void NuOsc::get_flux(Flux * RESTRICT out_flux, const real *in_field, const int stride)
{
    /*
     *    7th order WENO reconstruction.
     *   ------------------------------
     *
     *   Inputs:
     *       - lflux & rflux
     *           - Type: array.
     *           - Description: holds the flux values for a given dimension
     *       - field
     *           - Type: array
     *           - Description: Holds the values of the field variable on the grid for a given dimension.
     *   Modifies:
     *       - out_flux->l2h & out_flux->h2l.
     */
    const real EPS = 1E-6;

    const real gamma0_l2h = 4. / 35.;
    const real gamma1_l2h = 18. / 35.;
    const real gamma2_l2h = 12. / 35.;
    const real gamma3_l2h = 1. / 35.;

    const real gamma0_h2l = 1. / 35.;
    const real gamma1_h2l = 12. / 35.;
    const real gamma2_h2l = 18. / 35.;
    const real gamma3_h2l = 4. / 35.;

    /*
     *   Smoothness Indices of the stencils.
     *   -----------------------------------
     *   SI0 -> for stensil S0 = {i, i+1, i+2, i+3} -> r = 0 left shift.
     *   SI1 -> for stensil S1 = {i-1, i, i+1, i+2} -> r = 1 left shift.
     *   SI2 -> for stensil S2 = {i-2, i-1, i, i+1} -> r = 2 left shift.
     *   SI2 -> for stensil S3 = {i-3, i-2, i-1, i} -> r = 3 left shift.
     */

    #pragma acc parallel loop independent collapse(2)
    #pragma omp parallel for
    for (int zid = -1; zid < nz + 1; zid++) {
    #pragma omp _SIMD_
    for (int bin = 0; bin < nv; bin++)    {
                    auto ijkv = idx(0, zid, bin);
                    const real *u = &in_field[ijkv];
                    int i_1 = -stride;
                    int i_2 = -2*stride;
                    int i_3 = -3*stride;
                    int i0 = 0;
                    int i1 = (1 * stride);
                    int i2 = (2 * stride);
                    int i3 = (3 * stride);
                    real SI0 = u[i0] * (2107 * u[i0] - 9402 * u[i1] + 7042 * u[i2] - 1854 * u[i3]) + u[i1] * (11003 * u[i1] - 17246 * u[i2] + 4642 * u[i3]) + u[i2] * (7043 * u[i2] - 3882 * u[i3]) + 547 * u[i3] * u[i3];
                    real SI1 = u[i_1] * (547 * u[i_1] - 2522 * u[i0] + 1922 * u[i1] - 494 * u[i2]) + u[i0] * (3443 * u[i0] - 5966 * u[i1] + 1602 * u[i2]) + u[i1] * (2843 * u[i1] - 1642 * u[i2]) + 267 * u[i2] * u[i2];
                    real SI2 = u[i_2] * (267 * u[i_2] - 1642 * u[i_1] + 1602 * u[i0] - 494 * u[i1]) + u[i_1] * (2843 * u[i_1] - 5966 * u[i0] + 1922 * u[i1]) + u[i0] * (3443 * u[i0] - 2522 * u[i1]) + 547 * u[i1] * u[i1];
                    real SI3 = u[i_3] * (547 * u[i_3] - 3882 * u[i_2] + 4642 * u[i_1] - 1854 * u[i0]) + u[i_2] * (7043 * u[i_2] - 17246 * u[i_1] + 7042 * u[i0]) + u[i_1] * (11003 * u[i_1] - 9402 * u[i0]) + 2107 * u[i0] * u[i0];
                    //-------------------------------------------------------- Flux: low to high --------------------------------------------------------//
                    {
                    real w0 = gamma0_l2h / pow(EPS + SI0, 2);
                    real w1 = gamma1_l2h / pow(EPS + SI1, 2);
                    real w2 = gamma2_l2h / pow(EPS + SI2, 2);
                    real w3 = gamma3_l2h / pow(EPS + SI3, 2);
                    real u0 = (1. / 4.) * u[i0] + (13. / 12.) * u[i1] - (5. / 12.) * u[i2] + (1. / 12.) * u[i3];       // r = 0
                    real u1 = (-1. / 12.) * u[i_1] + (7. / 12.) * u[i0] + (7. / 12.) * u[i1] - (1. / 12.) * u[i2];     // r = 1
                    real u2 = (1. / 12.) * u[i_2] - (5. / 12.) * u[i_1] + (13. / 12.) * u[i0] + (1. / 4.) * u[i1];     // r = 2
                    real u3 = (-1. / 4.) * u[i_3] + (13. / 12.) * u[i_2] - (23. / 12.) * u[i_1] + (25. / 12.) * u[i0]; // r = 3
                    out_flux->l2h[ijkv] = (w0 * u0 + w1 * u1 + w2 * u2 + w3 * u3)/(w0 + w1 + w2 + w3);
                    }
                    //-------------------------------------------------------- Flux: high to low --------------------------------------------------------//
                    {
                    real w0 = gamma0_h2l / pow(EPS + SI0, 2);
                    real w1 = gamma1_h2l / pow(EPS + SI1, 2);
                    real w2 = gamma2_h2l / pow(EPS + SI2, 2);
                    real w3 = gamma3_h2l / pow(EPS + SI3, 2);
                    real u0 = (-1. / 4.) * u[i3] + (13. / 12.) * u[i2] - (23. / 12.) * u[i1] + (25. / 12.) * u[i0]; // r = 0
                    real u1 = (1. / 12.) * u[i2] - (5. / 12.) * u[i1] + (13. / 12.) * u[i0] + (1. / 4.) * u[i_1];   // r = 1
                    real u2 = (-1. / 12.) * u[i1] + (7. / 12.) * u[i0] + (7. / 12.) * u[i_1] - (1. / 12.) * u[i_2]; // r = 2
                    real u3 = (1. / 4.) * u[i0] + (13. / 12.) * u[i_1] - (5. / 12.) * u[i_2] + (1. / 12.) * u[i_3]; // r = 3
                    out_flux->h2l[ijkv] = (w0 * u0 + w1 * u1 + w2 * u2 + w3 * u3)/(w0 + w1 + w2 + w3);
                    }
    } // end of bin
    } // end of xyz
}
#endif
