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
    #pragma omp parallel for collapse(2)
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

        // OMP for not useful here
        #pragma acc loop
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
            real ko_eps_z = -ko/dz/16.0;
  #ifdef COSENU2D
            real ko_eps_x = -ko/dx/16.0;
            #define KO_FD(x) ( ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] ) + \
                               ko_eps_x * ( x[-2*nzv] + x[2*nzv] - 4*(x[-nzv]+x[nzv]) + 6*x[0] ) )
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
            real factor_x = -vx[v]/(12*dx);
            #define ADV_FD(x) ( factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv]  ) ) + \
                                factor_x*(  (x[-2*nzv]-x[2*nzv]) - 8.0*(x[-nzv]-x[nzv] ) ) )
  #else
            #define ADV_FD(x)   factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv])  )
  #endif

#endif

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

            #if 0
            real RIee    =         exr[0]  *( -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI  ) +  exi[0]*( -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR           ) ;
            real RIexr   =   (xx[0]-ee[0]) *( -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI  ) +  exi[0]*( -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee ) ;
            real RIexi   =   (xx[0]-ee[0]) *( -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR  ) -  exr[0]*( -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee ) ;
            real RIbee   =        bexr[0]  *( -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI  ) - bexi[0]*( -vx[v]*ivxdv_bexR_m_exR          -vy[v]*ivydv_bexR_m_exR            ) ;
            real RIbexr  = (bxx[0]-bee[0]) *( -vx[v]*ivxdv_bexI_p_exI -vy[v]*ivydv_bexI_p_exI  ) - bexi[0]*( -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee ) ;
            real RIbexi  = (bee[0]-bxx[0]) *( -vx[v]*ivxdv_bexR_m_exR -vy[v]*ivydv_bexR_m_exR  ) + bexr[0]*( -vx[v]*ivxdv_bxx_m_bee_m_xx_p_ee -vy[v]*ivydv_bxx_m_bee_m_xx_p_ee) ;
            // for axi-symm case, the vx/vy terms is always less than 1e-4...
            if ( std::abs(RIee) >1e-4 ) printf("RIee: %d %d %g %g\n", i,j,sqrt(vx[v]*vx[v]+vy[v]*vy[v]), RIee);
            if ( std::abs(RIexi)>1e-4 ) printf("RIexi: %d %d %g %g\n",i,j, sqrt(vx[v]*vx[v]+vy[v]*vy[v]), RIexi);
            if ( std::abs(RIbee) >1e-4 ) printf("RIbee: %d %d %g %g\n",i,j, sqrt(vx[v]*vx[v]+vy[v]*vy[v]), RIbee);
            if ( std::abs(RIbexi)>1e-4 ) printf("RIbexi: %d %d %g %g\n",i,j, sqrt(vx[v]*vx[v]+vy[v]*vy[v]), RIbexi);
            #endif

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



