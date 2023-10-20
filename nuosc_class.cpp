#include "nuosc_class.h"
#include <numeric> // std::inner_product


real flimiter(real a, real b) {
	//return 0.5*(sgn(a)+sgn(b))*min(abs(a),abs(b));    // minmod
	return 0.5*(sgn(a)+sgn(b))*max( min(2*abs(a),abs(b)), min(abs(a),2*abs(b)) );   // superbee
}


void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("calRHS");
#endif

    // Just for shorthand
    auto vw = grid.vw.data();
    auto vx = grid.vx.data();
    auto vy = grid.vy.data();
    auto vz = grid.vz.data();
    const int nzv = grid.nv*(grid.nz+2*grid.gz);

#pragma omp parallel for collapse(2)
#pragma acc parallel loop independent collapse(2)
    for (int i=0;i<grid.nx; ++i)
    for (int j=0;j<grid.nz; ++j) {

        // common integral over vz'  // from 3*4 --> 8*4
        std::array<real,4> emRm{{0,0,0,0}};
        std::array<real,4> emIp{{0,0,0,0}};
        std::array<real,4> eemm{{0,0,0,0}};
        #ifdef N_FLAVOR_3
        std::array<real,4> mtRm{{0,0,0,0}};
        std::array<real,4> mtIp{{0,0,0,0}};
        std::array<real,4> teRm{{0,0,0,0}};
        std::array<real,4> teIp{{0,0,0,0}};
        std::array<real,4> mmtt{{0,0,0,0}};
        #endif

        // OMP reduction not useful here
#pragma acc loop reduction(+:idv_emRm[0:4],idv_emIp[0:4] )
        for (int k=0;k<grid.nv; ++k) {
            auto ijk = grid.idx(i,j,k);

	    const std::array<real,4> v4{1,vx[k],vy[k],vz[k]};
 
            #pragma unroll
            for(int s=0;s<4;++s) {
                emRm[s]   += v4[s]* vw[k]*(in->bemr[ijk] - in->emr[ijk] );
                emIp[s]   += v4[s]* vw[k]*(in->bemi[ijk] + in->emi[ijk] );
                eemm[s]   += v4[s]* vw[k]*(in->bee[ijk]-in->bmm[ijk]-in->ee[ijk]+in->mm[ijk] );
                #ifdef N_FLAVOR_3
                mtRm[s]   += v4[s]* vw[k]*(in->bmtr[ijk] - in->mtr[ijk] );
                mtIp[s]   += v4[s]* vw[k]*(in->bmti[ijk] + in->mti[ijk] );
                teRm[s]   += v4[s]* vw[k]*(in->bter[ijk] - in->ter[ijk] );
                teIp[s]   += v4[s]* vw[k]*(in->btei[ijk] + in->tei[ijk] );
                mmtt[s]   += v4[s]* vw[k]*(in->bmm[ijk]-in->btt[ijk]-in->mm[ijk]+in->tt[ijk] );
                //ttee[s]   = - eemm - mmtt;
                #endif
            }
            // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
        }

        // OMP for not useful here
#pragma acc loop
        for (int v=0;v<grid.nv; ++v) {

            auto ijv = grid.idx(i,j,v);

            // The base pointer for this stencil
            real *ee    = &(in->ee    [ijv]);
            real *mm    = &(in->mm    [ijv]);
            real *emr   = &(in->emr [ijv]);
            real *emi   = &(in->emi [ijv]);
            real *bee    = &(in->bee    [ijv]);
            real *bmm    = &(in->bmm    [ijv]);
            real *bemr   = &(in->bemr [ijv]);
            real *bemi   = &(in->bemi [ijv]);
            #ifdef N_FLAVOR_3
            real *tt    = &(in->tt  [ijv]);
            real *mtr   = &(in->mtr [ijv]);
            real *mti   = &(in->mti [ijv]);
            real *ter   = &(in->ter [ijv]);
            real *tei   = &(in->tei [ijv]);
            real *btt    = &(in->btt [ijv]);
            real *bmtr   = &(in->bmtr [ijv]);
            real *bmti   = &(in->bmti [ijv]);
            real *bter   = &(in->bter [ijv]);
            real *btei   = &(in->btei [ijv]);
            #endif

#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps_x = -ko/grid.dx/64.0;
            real ko_eps_z = -ko/grid.dz/64.0;
            #define KO_FD(x) ( ko_eps_z*( x[-3*grid.nv] + x[3*grid.nv] - 6*(x[-2*grid.nv] +x[2*grid.nv]) + 15*(x[-grid.nv] +x[grid.nv]) - 20*x[0] ) + \
                               ko_eps_x*( x[-3*nzv]     + x[3*nzv]     - 6*(x[-2*nzv]+x[2*nzv])          + 15*(x[-nzv]     +x[nzv])     - 20*x[0] ) )
#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps_x = -ko/grid.dx/16.0;
            real ko_eps_z = -ko/grid.dz/16.0;
            #define KO_FD(x) ( ko_eps_z * ( x[-2*grid.nv] + x[2*grid.nv] - 4*(x[-grid.nv] +x[grid.nv]) + 6*x[0] ) + \
                               ko_eps_x * ( x[-2*nzv]     + x[2*nzv]     - 4*(x[-nzv]+x[nzv])          + 6*x[0] ) )
#endif

//#define ADVEC_MUSCL
//#define ADVEC_UPWIND2O
//#define ADVEC_LOPESIDE
//#define ADVEC_CFD2O
#ifdef ADVEC_OFF
             #define ADV_FD(x)     (0.0)
#elif defined(ADVEC_MUSCL)
	    // For advection eq: F(u) = v*(uL*int(v>0)  + uR*int(v<0))
/*
	    real uxLp = x[0]       + 0.5* minmod(x[0]       -x[-nzv],      x[nzv]    -x[0])       * (x[nzv]    -x[0]);
	    real uzLp = x[0]       + 0.5* minmod(x[0]       -x[-grid.nv],  x[grid.nv]-x[0])       * (x[grid.nv]-x[0]);
	    real uxLm = x[-nzv]    + 0.5* minmod(x[-nzv]    -x[-2*nzv],    x[0]      -x[-nzv])    * (x[0]      -x[-nzv]);
	    real uzLm = x[-grid.nv]+ 0.5* minmod(x[-grid.nv]-x[-2*grid.nv],x[0]      -x[-grid.nv])* (x[0]      -x[-grid.nv]);
	    real uxRp = x[nzv]     - 0.5* minmod(x[nzv]    -x[0],       x[2*nzv]    -x[nzv])    * (x[2*nzv]    -x[nzv]);
	    real uzRp = x[grid.nv] - 0.5* minmod(x[grid.nv]-x[0],       x[2*grid.nv]-x[grid.nv])* (x[2*grid.nv]-x[grid.nv]);
	    real uxRm = x[0]       - 0.5* minmod(x[0]      -x[-nzv],    x[nzv]      -x[0])      * (x[nzv]      -x[0]);
	    real uzRm = x[0]       - 0.5* minmod(x[0]      -x[-grid.nv],x[grid.nv]  -x[0])      * (x[grid.nv]  -x[0]);
            #define ADV_FD(x)  ( -grid.vx[v]/(grid.dx)* ( (uxLp-uxLm)*int(grid.vx[v]>0) + (uxRp-uxRm)*int(grid.vx[v]<0) )   \
                                 -grid.vz[v]/(grid.dz)* ( (uzLp-uzLm)*int(grid.vz[v]>0) + (uzRp-uzRm)*int(grid.vz[v]<0) ) )
*/
            #define ADV_FD(x)  ( -grid.vx[v]/(grid.dx)* ( int(grid.vx[v]>0)*  \
    ( x[0]       + 0.5* flimiter(x[0]       -x[-nzv],      x[nzv]    -x[0])      \
    - x[-nzv]   - 0.5* flimiter(x[-nzv]    -x[-2*nzv],    x[0]      -x[-nzv])   ) \
               + int(grid.vx[v]<0)* \
    ( x[nzv]     - 0.5* flimiter(x[nzv]    -x[0],       x[2*nzv]    -x[nzv])   \
      - x[0]     + 0.5* flimiter(x[0]      -x[-nzv],    x[nzv]      -x[0])    )   )   \
                                 -grid.vz[v]/(grid.dz)* ( int(grid.vz[v]>0)* \
    ( x[0]       + 0.5* flimiter(x[0]       -x[-grid.nv],  x[grid.nv]-x[0])     \
     -x[-grid.nv]- 0.5* flimiter(x[-grid.nv]-x[-2*grid.nv],x[0]      -x[-grid.nv])  ) \
              + int(grid.vz[v]<0)* \
     (  x[grid.nv] - 0.5* flimiter(x[grid.nv]-x[0],       x[2*grid.nv]-x[grid.nv])   \
        - x[0]       - 0.5* flimiter(x[0]      -x[-grid.nv],x[grid.nv]  -x[0])    )  ) )

#elif defined(ADVEC_UPWIND2O)
            int sx = sgn(grid.vx[v]);
            int sz = sgn(grid.vz[v]);
            real factor_x = -sx*grid.vx[v]/(2*grid.dx);
            real factor_z = -sz*grid.vz[v]/(2*grid.dz);
            #define ADV_FD(x)  ( factor_z*( x[-2*sz*grid.nv] - 4*x[-sz*grid.nv] + 3*x[0] ) + \
                                 factor_x*( x[-2*sx*nzv]     - 4*x[-sx*nzv]     + 3*x[0] ) )
#elif defined(ADVEC_LOPESIDE)
            // advection term: (4-th order lopsided finite differencing)
            int sx = sgn(grid.vx[v]);
            int sz = sgn(grid.vz[v]);
            real factor_x = -sx*grid.vx[v]/(12*grid.dx);
            real factor_z = -sz*grid.vz[v]/(12*grid.dz);
            #define ADV_FD(x)  ( factor_z* ( -x[-3*sz*grid.nv] + 6*x[-2*sz*grid.nv] - 18*x[-sz*grid.nv] + 10*x[0] + 3*x[sz*grid.nv] )   + \
                                 factor_x* ( -x[-3*sx*nzv]     + 6*x[-2*sx*nzv]     - 18*x[-sx*nzv]     + 10*x[0] + 3*x[sx*nzv]     ) )
#elif defined(ADVEC_CFD2O)
            real factor_x = -grid.vx[v]/(2*grid.dx);
            real factor_z = -grid.vz[v]/(2*grid.dz);
            #define ADV_FD(x) ( factor_z*( (x[grid.nv] -x[-grid.nv]  ) ) + \
                                factor_x*( (x[nzv]     -x[-nzv]      ) ) )
#else
            real factor_x = -grid.vx[v]/(12*grid.dx);
            real factor_z = -grid.vz[v]/(12*grid.dz);
            #define ADV_FD(x) ( factor_z*( (x[-2*grid.nv] - x[2*grid.nv]) - 8.0*(x[-grid.nv] - x[grid.nv] ) ) + \
                                factor_x*( (x[-2*nzv]     - x[2*nzv])     - 8.0*(x[-nzv]     - x[nzv]     ) ) )
#endif


	    #define INP(x) (x[0]-x[1]*vx[v]-x[2]*vy[v]-x[3]*vz[v])

#ifndef N_FLAVOR_3
	    auto iemRm = INP(emRm);
	    auto iemIp = INP(emIp);
	    auto ieemm = INP(eemm);
            real Iee   = 2*mu*( emi[0]*iemRm + emr[0]*iemIp  );
            real Imm   = -Iee;
	    real Iemr  =  -mu*( (ee[0]-mm[0])*iemIp + ieemm*emi[0] );
	    real Iemi  =   mu*( (mm[0]-ee[0])*iemRm + ieemm*emr[0] );
            real Ibee  = 2*mu*( bemi[0]*iemRm + bemr[0]*iemIp );
            real Ibmm  = -Ibee;
	    real Ibemr =  -mu*( (bee[0]-bmm[0])*iemIp + INP(eemm)*bemi[0] );
	    real Ibemi =   mu*( (bmm[0]-bee[0])*iemRm + INP(eemm)*bemr[0] );
#else
	    auto iemRm = INP(emRm);
	    auto iemIp = INP(emIp);
	    auto ieemm = INP(eemm);
	    auto immtt = INP(mmtt);
	    auto ittee = -(ieemm + immtt);

            #ifdef 0
	    auto imtRm = INP(mtRm);
	    auto imtIp = INP(mtIp);
	    auto iteRm = INP(teRm);
	    auto iteIp = INP(teIp);
            real Iee  = 2*mu*( emi[0]*iemRm + emr[0]*iemIp - tei[0]*iteRm - ter[0]*iteIp );
            real Imm  = 2*mu*(-emi[0]*iemRm - emr[0]*iemIp + mti[0]*imtRm + mtr[0]*imtIp );
            real Itt  = 2*mu*(-mti[0]*imtRm - mtr[0]*imtIp + tei[0]*iteRm + ter[0]*iteIp );
	    real Iemr =  -mu*((ee[0]-mm[0])*iemIp + ieemm*emi[0] - imtIp*ter[0] + imtRm*tei[0] - mti[0]*iteRm + mtr[0]*iteIp );
	    real Iemi =   mu*((mm[0]-ee[0])*iemRm + ieemm*emr[0] - imtIp*tei[0] - imtRm*ter[0] + mti[0]*iteIp + mtr[0]*iteRm );
	    real Imtr =  -mu*((mm[0]-tt[0])*imtIp + iemIp*ter[0] - iemRm*tei[0] + emi[0]*iteRm - emr[0]*iteIp + mti[0]*immtt );
	    real Imti =   mu*((tt[0]-mm[0])*imtRm + iemIp*tei[0] + iemRm*ter[0] - emi[0]*iteIp - emr[0]*iteRm + mtr[0]*immtt );
	    real Iter =  -mu*((tt[0]-ee[0])*iteIp - iemIp*mtr[0] + iemRm*mti[0] - emi[0]*imtRm + emr[0]*imtIp + tei[0]*ittee );
	    real Itei =   mu*((ee[0]-tt[0])*iteRm - iemIp*mti[0] - iemRm*mtr[0] + emi[0]*imtIp + emr[0]*imtRm + ter[0]*ittee );

            real Ibee  = 2*mu*( bemi[0]*iemRm + bemr[0]*iemIp - btei[0]*iteRm - bter[0]*iteIp );
            real Ibmm  = 2*mu*(-bemi[0]*iemRm - bemr[0]*iemIp + bmti[0]*imtRm + bmtr[0]*imtIp );
            real Ibtt  = 2*mu*(-bmti[0]*imtRm - bmtr[0]*imtIp + btei[0]*iteRm + bter[0]*iteIp );
	    real Ibemr =  -mu*((bee[0]-bmm[0])*iemIp + ieemm*bemi[0] - imtIp*bter[0] + imtRm*btei[0] - bmti[0]*iteRm + bmtr[0]*iteIp );
	    real Ibemi =   mu*((bmm[0]-bee[0])*iemRm + ieemm*bemr[0] - imtIp*btei[0] - imtRm*bter[0] + bmti[0]*iteIp + bmtr[0]*iteRm );
	    real Ibmtr =  -mu*((bmm[0]-btt[0])*imtIp + iemIp*bter[0] - iemRm*btei[0] + bemi[0]*iteRm - bemr[0]*iteIp + bmti[0]*immtt );
	    real Ibmti =   mu*((btt[0]-bmm[0])*imtRm + iemIp*btei[0] + iemRm*bter[0] - bemi[0]*iteIp - bemr[0]*iteRm + bmtr[0]*immtt );
	    real Ibter =  -mu*((btt[0]-bee[0])*iteIp - iemIp*bmtr[0] + iemRm*bmti[0] - bemi[0]*imtRm + bemr[0]*imtIp + btei[0]*ittee );
	    real Ibtei =   mu*((bee[0]-btt[0])*iteRm - iemIp*bmti[0] - iemRm*bmtr[0] + bemi[0]*imtIp + bemr[0]*imtRm + bter[0]*ittee );
            #else
            real Iee  = 2*mu*( emi[0]*INP(emRm) + emr[0]*INP(emIp) - tei[0]*INP(teRm) - ter[0]*INP(teIp) );
            real Imm  = 2*mu*(-emi[0]*INP(emRm) - emr[0]*INP(emIp) + mti[0]*INP(mtRm) + mtr[0]*INP(mtIp) );
            real Itt  = 2*mu*(-mti[0]*INP(mtRm) - mtr[0]*INP(mtIp) + tei[0]*INP(teRm) + ter[0]*INP(teIp) );
	    real Iemr =  -mu*((ee[0]-mm[0])*INP(emIp) + INP(eemm)*emi[0] - INP(mtIp)*ter[0] + INP(mtRm)*tei[0] - mti[0]*INP(teRm) + mtr[0]*INP(teIp) );
	    real Iemi =   mu*((mm[0]-ee[0])*INP(emRm) + INP(eemm)*emr[0] - INP(mtIp)*tei[0] - INP(mtRm)*ter[0] + mti[0]*INP(teIp) + mtr[0]*INP(teRm) );
	    real Imtr =  -mu*((mm[0]-tt[0])*INP(mtIp) + INP(emIp)*ter[0] - INP(emRm)*tei[0] + emi[0]*INP(teRm) - emr[0]*INP(teIp) + mti[0]*INP(mmtt) );
	    real Imti =   mu*((tt[0]-mm[0])*INP(mtRm) + INP(emIp)*tei[0] + INP(emRm)*ter[0] - emi[0]*INP(teIp) - emr[0]*INP(teRm) + mtr[0]*INP(mmtt) );
	    real Iter =  -mu*((tt[0]-ee[0])*INP(teIp) - INP(emIp)*mtr[0] + INP(emRm)*mti[0] - emi[0]*INP(mtRm) + emr[0]*INP(mtIp) + tei[0]*ittee );
	    real Itei =   mu*((ee[0]-tt[0])*INP(teRm) - INP(emIp)*mti[0] - INP(emRm)*mtr[0] + emi[0]*INP(mtIp) + emr[0]*INP(mtRm) + ter[0]*ittee );

            real Ibee  = 2*mu*( bemi[0]*INP(emRm) + bemr[0]*INP(emIp) - btei[0]*INP(teRm) - bter[0]*INP(teIp) );
            real Ibmm  = 2*mu*(-bemi[0]*INP(emRm) - bemr[0]*INP(emIp) + bmti[0]*INP(mtRm) + bmtr[0]*INP(mtIp) );
            real Ibtt  = 2*mu*(-bmti[0]*INP(mtRm) - bmtr[0]*INP(mtIp) + btei[0]*INP(teRm) + bter[0]*INP(teIp) );
	    real Ibemr =  -mu*((bee[0]-bmm[0])*INP(emIp) + INP(eemm)*bemi[0] - INP(mtIp)*bter[0] + INP(mtRm)*btei[0] - bmti[0]*INP(teRm) + bmtr[0]*INP(teIp) );
	    real Ibemi =   mu*((bmm[0]-bee[0])*INP(emRm) + INP(eemm)*bemr[0] - INP(mtIp)*btei[0] - INP(mtRm)*bter[0] + bmti[0]*INP(teIp) + bmtr[0]*INP(teRm) );
	    real Ibmtr =  -mu*((bmm[0]-btt[0])*INP(mtIp) + INP(emIp)*bter[0] - INP(emRm)*btei[0] + bemi[0]*INP(teRm) - bemr[0]*INP(teIp) + bmti[0]*INP(mmtt) );
	    real Ibmti =   mu*((btt[0]-bmm[0])*INP(mtRm) + INP(emIp)*btei[0] + INP(emRm)*bter[0] - bemi[0]*INP(teIp) - bemr[0]*INP(teRm) + bmtr[0]*INP(mmtt) );
	    real Ibter =  -mu*((btt[0]-bee[0])*INP(teIp) - INP(emIp)*bmtr[0] + INP(emRm)*bmti[0] - bemi[0]*INP(mtRm) + bemr[0]*INP(mtIp) + btei[0]*ittee );
	    real Ibtei =   mu*((bee[0]-btt[0])*INP(teRm) - INP(emIp)*bmti[0] - INP(emRm)*bmtr[0] + bemi[0]*INP(mtIp) + bemr[0]*INP(mtRm) + bter[0]*ittee );
            #endif
#endif  // N_FLAVOR_3

	    #undef INP

#ifndef VACUUM_OFF
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...

            #ifdef N_FLAVOR_3
            out->ee  [ijv] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo * 2*(emi[0]*hemr + emr[0]*hemi - htei*ter[0] + hter*tei[0]);
            out->mm  [ijv] =  Imm   + ADV_FD(mm)   + KO_FD(mm)   + pmo * 2*(emi[0]*hemr - emr[0]*hemi + hmti*mtr[0] - hmtr*mti[0]);
            out->emr [ijv] =  Iemr  + ADV_FD(emr)  + KO_FD(emr)  + pmo*( (mm[0]-ee[0])*hemi + emi[0]*(hee - hmm) + hmti*ter[0] + hmtr*tei[0] - htei*mtr[0] - hter*mti[0] );
            out->emi [ijv] =  Iemi  + ADV_FD(emi)  + KO_FD(emi)  + pmo*( (ee[0]-mm[0])*hemr - emr[0]*(hee - hmm) - hmti*tei[0] + hmtr*ter[0] + htei*mti[0] - hter*mtr[0] );
            out->bee [ijv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo * 2*(bemi[0]*hemr + bemr[0]*hemi - htei*bter[0] + hter*btei[0]);
            out->bmm [ijv] =  Ibmm  + ADV_FD(bmm)  + KO_FD(bmm)  + pmo * 2*(bemi[0]*hemr - bemr[0]*hemi + hmti*bmtr[0] - hmtr*bmti[0]);
            out->bemr[ijv] =  Ibemr + ADV_FD(bemr) + KO_FD(bemr) + pmo*( (bmm[0]-bee[0])*hemi + bemi[0]*(hee - hmm) + hmti*bter[0] + hmtr*btei[0] - htei*bmtr[0] - hter*bmti[0] );
            out->bemi[ijv] =  Ibemi + ADV_FD(bemi) + KO_FD(bemi) + pmo*( (bee[0]-bmm[0])*hemr - bemr[0]*(hee - hmm) - hmti*btei[0] + hmtr*bter[0] + htei*bmti[0] - hter*bmtr[0] );
            out->tt  [ijv] =   Itt + ADV_FD(tt)   + KO_FD(tt)   - pmo* 2*(hmti*mtr[0] + hmtr*mti[0] + htei*ter[0] - hter*tei[0]);;
            out->mtr [ijv] =  Imtr + ADV_FD(mtr)  + KO_FD(mtr)  + pmo* ( (tt[0]-mm[0])*hmti + emr[0]*htei + emi[0]*hter - hemi*ter[0] - hemr*tei[0] + mti[0]*(hmm - htt) );
            out->mti [ijv] =  Imti + ADV_FD(mti)  + KO_FD(mti)  + pmo* ( (mm[0]-tt[0])*hmtr + emr[0]*hter - emi[0]*htei + hemi*tei[0] - hemr*ter[0] - mtr[0]*(hmm - htt) );
            out->ter [ijv] =  Iter + ADV_FD(ter)  + KO_FD(ter)  + pmo* ( (ee[0]-tt[0])*htei - emr[0]*hmti - emi[0]*hmtr + hemi*mtr[0] + hemr*mti[0] - tei[0]*(hee - htt) );
            out->tei [ijv] =  Itei + ADV_FD(tei)  + KO_FD(tei)  + pmo* ( (tt[0]-ee[0])*hter - emr[0]*hmtr + emi[0]*hmti - hemi*mti[0] + hemr*mtr[0] + ter[0]*(hee - htt) );
            out->btt [ijv] =  Ibtt + ADV_FD(btt)  + KO_FD(btt)  - pmo* 2*(hmti*mtr[0] + hmtr*mti[0] + htei*ter[0] - hter*tei[0]);;
            out->bmtr[ijv] = Ibmtr + ADV_FD(bmtr) + KO_FD(bmtr) + pmo* ( (btt[0]-bmm[0])*hmti + bemr[0]*htei + bemi[0]*hter - hemr*btei[0] - hemi*bter[0] + bmti[0]*(hmm - htt) );
            out->bmti[ijv] = Ibmti + ADV_FD(bmti) + KO_FD(bmti) + pmo* ( (bmm[0]-btt[0])*hmtr + bemr[0]*hter - bemi[0]*htei - hemr*bter[0] + hemi*btei[0] - bmtr[0]*(hmm - htt) );
            out->bter[ijv] = Ibter + ADV_FD(bter) + KO_FD(bter) + pmo* ( (bee[0]-btt[0])*htei - bemr[0]*hmti - bemi[0]*hmtr + hemi*bmtr[0] + hemr*bmti[0] - btei[0]*(hee - htt) );
            out->btei[ijv] = Ibtei + ADV_FD(btei) + KO_FD(btei) + pmo* ( (btt[0]-bee[0])*hter - bemr[0]*hmtr + bemi[0]*hmti - hemi*bmti[0] + hemr*bmtr[0] + bter[0]*(hee - htt) );
            #else
            out->ee  [ijv] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo* 2*st*emi [0];   //hemr = st
            out->mm  [ijv] =  Imm   + ADV_FD(mm)   + KO_FD(mm)   + pmo* 2*st*emi [0];
            out->emr [ijv] =  Iemr  + ADV_FD(emr)  + KO_FD(emr)  - pmo* 2*ct*emi [0];
            out->emi [ijv] =  Iemi  + ADV_FD(emi)  + KO_FD(emi)  + pmo*(2*ct*emr [0] + st*( ee[0] - mm[0] ) );
            out->bee [ijv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo* 2*st*bemi[0];
            out->bmm [ijv] = -Ibee  + ADV_FD(bmm)  + KO_FD(bmm)  + pmo* 2*st*bemi[0];
            out->bemr[ijv] =  Ibemr + ADV_FD(bemr) + KO_FD(bemr) - pmo* 2*ct*bemi[0];
            out->bemi[ijv] =  Ibemi + ADV_FD(bemi) + KO_FD(bemi) + pmo*(2*ct*bemr[0] + st*( bee[0] - bmm[0] ) );
            #endif

#else
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee  [ijv] =  Iee   + ADV_FD(ee)   + KO_FD(ee);
            out->mm  [ijv] = -Iee   + ADV_FD(mm)   + KO_FD(mm);
            out->emr [ijv] =  Iemr  + ADV_FD(emr)  + KO_FD(emr);
            out->emi [ijv] =  Iemi  + ADV_FD(emi)  + KO_FD(emi);
            out->bee [ijv] =  Ibee  + ADV_FD(bee)  + KO_FD(bee);
            out->bmm [ijv] = -Ibee  + ADV_FD(bmm)  + KO_FD(bmm);
            out->bemr[ijv] =  Ibemr + ADV_FD(bemr) + KO_FD(bemr);
            out->bemi[ijv] =  Ibemi + ADV_FD(bemi) + KO_FD(bemi);
#endif
        }
    }

#ifdef NVTX
    nvtxRangePop();
#endif
}




/* v0 = v1 + a * v2 */
inline void NuOsc::vectorize(FieldVar* const __restrict v0, FieldVar * const __restrict v1, const real a, FieldVar * const __restrict v2) {
#ifdef NVTX
    nvtxRangePush("vectorize");
#endif

    std::vector<real*> fv0 = v0->getAllFields();
    std::vector<real*> fv1 = v1->getAllFields();
    std::vector<real*> fv2 = v2->getAllFields();

    PARFORALL(i,j,v) {
        auto k = grid.idx(i,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fv0.at(f)[k] = fv1.at(f)[k] + a * fv2.at(f)[k];
/*
        v0->ee  [k] = v1->ee  [k] + a * v2->ee  [k];
        v0->mm  [k] = v1->mm  [k] + a * v2->mm  [k];
        v0->emr [k] = v1->emr [k] + a * v2->emr [k];
        v0->emi [k] = v1->emi [k] + a * v2->emi [k];
        v0->bee [k] = v1->bee [k] + a * v2->bee [k];
        v0->bmm [k] = v1->bmm [k] + a * v2->bmm [k];
        v0->bemr[k] = v1->bemr[k] + a * v2->bemr[k];
        v0->bemi[k] = v1->bemi[k] + a * v2->bemi[k];
        #ifdef FALVOR3
        v0->tt  [k] = v1->tt  [k] + a * v2->tt  [k];
        v0->mtr [k] = v1->mtr [k] + a * v2->mtr [k];
        v0->mti [k] = v1->mti [k] + a * v2->mti [k];
        v0->ter [k] = v1->ter [k] + a * v2->ter [k];
        v0->tei [k] = v1->tei [k] + a * v2->tei [k];
        v0->btt [k] = v1->btt [k] + a * v2->btt [k];
        v0->bmtr[k] = v1->bmtr[k] + a * v2->bmtr[k];
        v0->bmti[k] = v1->bmti[k] + a * v2->bmti[k];
        v0->bter[k] = v1->bter[k] + a * v2->bter[k];
        v0->btei[k] = v1->btei[k] + a * v2->btei[k];
        #endif
*/
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
inline void NuOsc::vectorize(FieldVar * const __restrict v0, FieldVar * const __restrict v1, const real a, FieldVar * const  __restrict v2, FieldVar * const  __restrict v3) {
#ifdef NVTX
    nvtxRangePush("vectorize");
#endif

    std::vector<real*> fv0 = v0->getAllFields();
    std::vector<real*> fv1 = v1->getAllFields();
    std::vector<real*> fv2 = v2->getAllFields();
    std::vector<real*> fv3 = v3->getAllFields();

    PARFORALL(i,j,v) {
        auto k = grid.idx(i,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fv0.at(f)[k] = fv1.at(f)[k] + a * (fv2.at(f)[k] + fv3.at(f)[k]);
/*
        v0->mm  [k] = v1->mm  [k] + a * (v2->mm  [k] + v3->mm  [k]);
        v0->emr [k] = v1->emr [k] + a * (v2->emr [k] + v3->emr [k]);
        v0->emi [k] = v1->emi [k] + a * (v2->emi [k] + v3->emi [k]);
        v0->bee [k] = v1->bee [k] + a * (v2->bee [k] + v3->bee [k]);
        v0->bmm [k] = v1->bmm [k] + a * (v2->bmm [k] + v3->bmm [k]);
        v0->bemr[k] = v1->bemr[k] + a * (v2->bemr[k] + v3->bemr[k]);
        v0->bemi[k] = v1->bemi[k] + a * (v2->bemi[k] + v3->bemi[k]);
        #ifdef FALVOR3
        v0->tt  [k] = v1->tt  [k] + a * (v2->tt  [k] + v3->tt  [k]);
        v0->mtr [k] = v1->mtr [k] + a * (v2->mtr [k] + v3->mtr [k]);
        v0->mti [k] = v1->mti [k] + a * (v2->mti [k] + v3->mti [k]);
        v0->ter [k] = v1->ter [k] + a * (v2->ter [k] + v3->ter [k]);
        v0->tei [k] = v1->tei [k] + a * (v2->tei [k] + v3->tei [k]);
        v0->btt  [k] = v1->btt  [k] + a * (v2->btt  [k] + v3->btt  [k]);
        v0->bmtr [k] = v1->bmtr [k] + a * (v2->bmtr [k] + v3->bmtr [k]);
        v0->bmti [k] = v1->bmti [k] + a * (v2->bmti [k] + v3->bmti [k]);
        v0->bter [k] = v1->bter [k] + a * (v2->bter [k] + v3->bter [k]);
        v0->btei [k] = v1->btei [k] + a * (v2->btei [k] + v3->btei [k]);
        #endif
*/
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = a*v1 + b*(v2 + dt*v3)
inline void NuOsc::vectorize(FieldVar* const __restrict v0, const real a, FieldVar * const __restrict v1, const real b, FieldVar* const __restrict v2, const real dt, FieldVar * const __restrict v3) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    std::vector<real*> fv0 = v0->getAllFields();
    std::vector<real*> fv1 = v1->getAllFields();
    std::vector<real*> fv2 = v2->getAllFields();
    std::vector<real*> fv3 = v3->getAllFields();

    PARFORALL(i,j,v) {
        auto k = grid.idx(i,j,v);
        #pragma unroll
        for (int f=0; f<nvar; ++f) fv0.at(f)[k] = a*fv1.at(f)[k] + b * (fv2.at(f)[k] + dt*fv3.at(f)[k]);
/*
        v0->mm  [k] = a*v1->mm  [k] + b * (v2->mm  [k] + dt*v3->mm  [k]);
        v0->emr [k] = a*v1->emr [k] + b * (v2->emr [k] + dt*v3->emr [k]);
        v0->emi [k] = a*v1->emi [k] + b * (v2->emi [k] + dt*v3->emi [k]);
        v0->bee [k] = a*v1->bee [k] + b * (v2->bee [k] + dt*v3->bee [k]);
        v0->bmm [k] = a*v1->bmm [k] + b * (v2->bmm [k] + dt*v3->bmm [k]);
        v0->bemr[k] = a*v1->bemr[k] + b * (v2->bemr[k] + dt*v3->bemr[k]);
        v0->bemi[k] = a*v1->bemi[k] + b * (v2->bemi[k] + dt*v3->bemi[k]);
        #ifdef FALVOR3
        v0->tt  [k] = a*v1->tt  [k] + b * (v2->tt  [k] + dt*v3->tt  [k]);
        v0->mtr [k] = a*v1->mtr [k] + b * (v2->mtr [k] + dt*v3->mtr [k]);
        v0->mti [k] = a*v1->mti [k] + b * (v2->mti [k] + dt*v3->mti [k]);
        v0->ter [k] = a*v1->ter [k] + b * (v2->ter [k] + dt*v3->ter [k]);
        v0->tei [k] = a*v1->tei [k] + b * (v2->tei [k] + dt*v3->tei [k]);
        v0->btt [k] = a*v1->btt [k] + b * (v2->btt [k] + dt*v3->btt [k]);
        v0->bmtr[k] = a*v1->bmtr[k] + b * (v2->bmtr[k] + dt*v3->bmtr[k]);
        v0->bmti[k] = a*v1->bmti[k] + b * (v2->bmti[k] + dt*v3->bmti[k]);
        v0->bter[k] = a*v1->bter[k] + b * (v2->bter[k] + dt*v3->bter[k]);
        v0->btei[k] = a*v1->btei[k] + b * (v2->btei[k] + dt*v3->btei[k]);
        #endif
*/
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

void NuOsc::step_srk3() {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    //Step-1
    calRHS(v_rhs, v_stat);   // L(u0)
    vectorize(v_pre, v_stat, dt, v_rhs);  // u1 = u0 + dt*L(u0)

    //Step-2
    calRHS(v_rhs, v_pre);   // L(u1)
    vectorize(v_pre, 3.0/4.0, v_stat, 1.0/4.0, v_pre, dt, v_rhs);   // u2

    //Step-3
    calRHS(v_cor, v_pre);   // L(u2)
    vectorize(v_stat, 1.0/3.0, v_stat, 2.0/3.0, v_pre, dt, v_cor);

    if(renorm) renormalize(v_stat);

    phy_time += dt;
#ifdef NVTX
    nvtxRangePop();
#endif
}

NuOsc::NuOsc(const int px_, const int pz_, const int nv_, const int nphi_, const int gx_,const int gz_,
        const real  x0_, const real  x1_, const real  z0_, const real  z1_, const real dx_, const real dz_, 
        const real CFL_, const real  ko_) : phy_time(0.), ko(ko_), dx(dx_), dz(dz_), 
        grid(px_,pz_,nv_,nphi_,gx_,gz_,x0_,x1_,z0_,z1_,dx_,dz_) {

        auto size = grid.get_lpts();
        real mem_per_var = size*8/1024./1024./1024;
        ds_L = dx_*dz_/(z1_-z0_)/(x1_-x0_);

        CFL = CFL_;
        dt = dz*CFL;

#ifdef COSENU_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

#endif
 
    //MPI_Barrier(MPI_COMM_WORLD);

    if (myrank==0) {
            printf("\nNuOsc2D on %d MPI ranks: %d core per rank.\n", grid.ranks, omp_get_max_threads() );
            printf("   Domain:  v: nv = %5d  nphi = %5d  on S2.\n", grid.get_nv(), grid.get_nphi() );
            printf("            x:( %12f %12f )  dx = %g\n", x0_,x1_, dx);
            printf("            z:( %12f %12f )  dz = %g\n", z0_,z1_, dz);
            printf("   Fieldvar: %d -- Size per var per rank = %.2f GB, totol memory per rank ~ %.2f GB\n", NVAR, mem_per_var, mem_per_var*50);
            printf("   dt = %g     CFL = %g\n", dt, CFL);
#ifdef BC_PERI
            printf("   Boundary: Periodic\n");
#else
            printf("   Boundary: Open\n");
#endif

#if defined(IM_V2D_POLAR_GRID)
            printf("   Velocity grid: uniform z-/phi-\n");
#else
            printf("   Velocity grid: Icosahedron sphere grid\n");
#endif

#ifndef KO_ORD_3
            printf("   Dissipation: 5-th order KO w/ eps = %g\n", ko);
#else
            printf("   Dissipation: 3-th order KO w/ eps = %g\n", ko);
#endif

#ifndef ADVEC_OFF
            printf("   Advection: ON (Center-FD)\n");
            //printf("   Use upwinded for advaction. (EXP. Always blowup!!\n");
            //printf("   Use lopsided FD for advaction\n");
#else
            printf("   Advection: OFF\n");
#endif

#ifdef VACUUM_OFF
            printf("   Vacuum term: OFF\n");
#else
            printf("   Vacuum term: pmo= %g theta= %g\n", pmo, theta);
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
            anafile << "### [ phy_time,   1:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0    7:Lem   8:ELNe]" << endl;
        }

	{   // Hvac
#ifdef N_FLAVOR_3
            const real dms12=7.39e-5*1267.*2., dms13=2.5229e-3* 1267.*2., theta12=33.82/180., theta13=8.61/180., theta23=48.3/180., theta_cp=0.;
            const real c12=cos(theta12), s12=sin(theta12), c13=cos(theta13), s13=sin(theta13), c23=cos(theta23), s23=sin(theta23);
            const real Ue2 = s12*c13;
            const real Um2r= c12*c23-s12*s13*s23*cos(theta_cp);
            const real Ut2r=-c12*s23-s12*s13*c23*cos(theta_cp);
            const real Um2i=        -s12*s13*s23*sin(theta_cp);
            const real Ut2i=        -s12*s13*c23*sin(theta_cp);
            const real Ue3r= s13*cos(theta_cp);
            const real Ue3i=-s13*sin(theta_cp);
            const real Um3 = c13*s23;
            const real Ut3 = c13*c23;

            hee  = dms12*(Ue2*Ue2)            +dms13*(s13*s13);
            hmm  = dms12*(Um2r*Um2r+Um2i*Um2i)+dms13*(Um3*Um3);
            htt  = dms12*(Ut2r*Ut2r+Ut2i*Ut2i)+dms13*(Ut3*Ut3);
            hemr = dms12*( Ue2*Um2r)          +dms13*(Ue3r*Um3);
            hemi = dms12*(-Ue2*Um2i)          +dms13*(Ue3i*Um3);
            hmtr = dms12*( Ue2*Ut2r)          +dms13*(Ue3r*Ut3);
            hmti = dms12*(-Ue2*Ut2i)          +dms13*(Ue3i*Ut3);
            hter = dms12*(Um2r*Ut2r+Um2i*Ut2i)+dms13*(Um3*Ut3);
            htei = dms12*(Ut2r*Um2i-Um2r*Ut2i);
            real Hvac_trace = (hee+hmm+htt)/3.0;
            hee -= Hvac_trace;
            hmm -= Hvac_trace;
            htt -= Hvac_trace;
#else
            const real theta = 37 * M_PI / 180.;  //1e-6;
            const real ct = cos(2*theta);
            const real st = sin(2*theta);
#endif // N_FLAVOR_2, N_FLAVOR_3
	}

    }

