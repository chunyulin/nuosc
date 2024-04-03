#include "nuosc_class.h"
#include "utils.h"

void NuOsc::calRHS(FieldVar * RESTRICT out, FieldVar * RESTRICT in) {
#ifdef PROFILE
nvtxRangePush("calRHS");
#endif

#ifdef COSENU_MPI
    pack_buffer(in);
    sync_launch();
    #ifdef NOT_OVERLAP
    waitall();
    unpack_buffer(in);
    calRHS_wo_bdry(out, in);
    #else
    calRHS_wo_bdry(out, in);
    waitall();
    unpack_buffer(in);
    #endif

    #ifndef ADVEC_OFF
    calRHS_with_bdry(out, in);
    #endif
#else
    updatePeriodicBoundary(in);
    calRHS_wo_bdry(out, in);
    #ifndef ADVEC_OFF
    calRHS_with_bdry(out, in);
    #endif
#endif
#ifdef PROFILE
nvtxRangePop();
#endif
}

void NuOsc::calRHS_with_bdry(FieldVar * RESTRICT out, const FieldVar * RESTRICT in) {
#ifdef PROFILE
    nvtxRangePush("calRHS_with_bdry");
#endif
    // for abbreviation
    const int nzv  = nv* (nx[2] + 2*gx[2]);
    const int nyzv = nzv*(nx[1] + 2*gx[1]);

    for (int f=0;f<nvar; ++f) {

#ifdef SCHEME_WENO7
        // adv-x
        get_flux(flux, in->wf[f], nyzv, 1, 0, 0);
        PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            int s = sgn(vx[v]);
            out->wf[f][ijkv] += -0.5*vx[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nyzv]) + std::abs(1-s)*(flux->h2l[ijkv+nyzv]-flux->h2l[ijkv]));
        }
        // adv-y
        get_flux(flux, in->wf[f], nzv, 0, 1, 0);
        PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            int s = sgn(vy[v]);
            out->wf[f][ijkv] += -0.5*vy[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nzv]) + std::abs(1-s)*(flux->h2l[ijkv+nzv]-flux->h2l[ijkv]));
        }
        // adv-z
        get_flux(flux, in->wf[f], nv, 0, 0, 1);
        PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            int s = sgn(vz[v]);
            out->wf[f][ijkv] += -0.5*vz[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nv]) + std::abs(1-s)*(flux->h2l[ijkv+nv]-flux->h2l[ijkv]));
        }
#else
        PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            const real *ff   = &(in->wf[f][ijkv]);
            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
    #ifdef SCHEME_FD8
            real factor_z = -vz[v]/(60*dx);
            real factor_y = -vy[v]/(60*dx);
            real factor_x = -vx[v]/(60*dx);
            #define ADV_FD(x) ( \
              factor_z*(-(x[-3*nv]  -x[3*nv])   + 9.0*(x[-2*nv]  -x[2*nv])   - 45.0*(x[-nv]  -x[nv])   ) + \
              factor_y*(-(x[-3*nzv] -x[3*nzv])  + 9.0*(x[-2*nzv] -x[2*nzv])  - 45.0*(x[-nzv] -x[nzv])  ) + \
              factor_x*(-(x[-3*nyzv]-x[3*nyzv]) + 9.0*(x[-2*nyzv]-x[2*nyzv]) - 45.0*(x[-nyzv]-x[nyzv]) ) )
    #elif FD6
            real factor_z = -vz[v]/(280*dx);
            real factor_y = -vy[v]/(280*dx);
            real factor_x = -vx[v]/(280*dx);
            #define ADV_FD(x) ( \
              factor_z*( (x[-4*nv]  -x[4*nv])   - (224./21.0)*(x[-3*nv]  -x[3*nv])   + 56.0*(x[-2*nv]  -x[2*nv])   - 224.0*(x[-nv]  -x[nv])   ) + \
              factor_y*( (x[-4*nzv] -x[4*nzv])  - (224./21.0)*(x[-3*nzv] -x[3*nzv])  + 56.0*(x[-2*nzv] -x[2*nzv])  - 224.0*(x[-nzv] -x[nzv])  ) + \
              factor_x*( (x[-4*nyzv]-x[4*nyzv]) - (224./21.0)*(x[-3*nyzv]-x[3*nyzv]) + 56.0*(x[-2*nyzv]-x[2*nyzv]) - 224.0*(x[-nyzv]-x[nyzv]) ) )
    #else
            real factor_z = -vz[v]/(12*dx);
            real factor_y = -vy[v]/(12*dx);
            real factor_x = -vx[v]/(12*dx);
            #define ADV_FD(x) ( \
              factor_z*(  (x[-2*nv]  -x[2*nv])   - 8.0*( x[-nv]  -x[nv]   ) ) + \
              factor_y*(  (x[-2*nzv] -x[2*nzv])  - 8.0*( x[-nzv] -x[nzv]  ) ) + \
              factor_x*(  (x[-2*nyzv]-x[2*nyzv]) - 8.0*( x[-nyzv]-x[nyzv] ) ) )
    #endif


            // prepare KO operator
            #ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps = -ko/dx/64.0;
            #define KO_FD(x) ko_eps*( \
               ( x[-3*nv]  +x[3*nv]   - 6.*(x[-2*nv]  +x[2*nv])   + 15.*(x[-nv]  + x[nv])  - 20.*x[0] ) + \
               ( x[-3*nzv] +x[3*nzv]  - 6.*(x[-2*nzv] +x[2*nzv])  + 15.*(x[-nzv] + x[nzv]) - 20.*x[0] ) + \
               ( x[-3*nyzv]+x[3*nyzv] - 6.*(x[-2*nyzv]+x[2*nyzv]) + 15.*(x[-nyzv]+ x[nyzv])- 20.*x[0] ) )
            #else
            // Kreiss-Oliger dissipation (3-nd order --> 4rd derivatives)
        #ifdef SCHEME_FD8
            // O(x^4) with 3 buffer zone
            real ko_eps = -ko/dx/16.0/6.0;
            #define KO_FD(x) ko_eps*( \
             ( -(x[-3*nv]  +x[3*nv])  + 12.0*(x[-2*nv]  +x[2*nv])  -39.0*(x[-nv]  +x[nv])  +56.0*x[0] ) + \
             ( -(x[-3*nzv] +x[3*nzv]) + 12.0*(x[-2*nzv] +x[2*nzv]) -39.0*(x[-nzv] +x[nzv]) +56.0*x[0] ) + \
             ( -(x[-3*nyzv]+x[3*nyzv])+ 12.0*(x[-2*nyzv]+x[2*nyzv])-39.0*(x[-nyzv]+x[nyzv])+56.0*x[0] ) )
        #elif FD6
            // O(x^6) with 4 buffer zone
            real ko_eps = -ko/dx/16.0/240.0;
            #define KO_FD(x) ko_eps*( \
             ( 7.0*(x[-4*nv]  +x[4*nv])  -96.0*(x[-3*nv]  +x[3*nv])  + 676.0*(x[-2*nv]  +x[2*nv])  -1952.0*(x[-nv]  +x[nv])  +2730.0*x[0] ) + \
             ( 7.0*(x[-4*nzv] +x[4*nzv]) -96.0*(x[-3*nzv] +x[3*nzv]) + 676.0*(x[-2*nzv] +x[2*nzv]) -1952.0*(x[-nzv] +x[nzv]) +2730.0*x[0] ) + \
             ( 7.0*(x[-4*nyzv]+x[4*nyzv])-96.0*(x[-3*nyzv]+x[3*nyzv])+ 676.0*(x[-2*nyzv]+x[2*nyzv])-1952.0*(x[-nyzv]+x[nyzv])+2730.0*x[0] ) )
        #else
            real ko_eps = -ko/dx/16.0;
            #define KO_FD(x) ko_eps*( \
               ( x[-2*nv]  +x[2*nv]   - 4.*(x[-nv]  +x[nv])   + 6.*x[0] ) + \
               ( x[-2*nzv] +x[2*nzv]  - 4.*(x[-nzv] +x[nzv])  + 6.*x[0] ) + \
               ( x[-2*nyzv]+x[2*nyzv] - 4.*(x[-nyzv]+x[nyzv]) + 6.*x[0] ) )
            #endif
        #endif

            out->wf[f][ijkv] += ADV_FD(ff) + KO_FD(ff);
        } // end for xyzv.
#endif // end of WENO7
    } // end for fields.

#ifdef PROFILE
    nvtxRangePop();
#endif
}

void NuOsc::calRHS_wo_bdry(FieldVar * RESTRICT out, const FieldVar * RESTRICT in) {
#ifdef PROFILE
    nvtxRangePush("calRHS_wo_bdry");
#endif
    #pragma acc parallel loop independent collapse(3)
    #pragma omp parallel for collapse(3)
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<nx[2]; ++k) {
        //
        // common integral factors over vz'
        //

#define IDEN(x) real x##0 = 0.0, x##1 = 0.0, x##2 = 0.0, x##3 = 0.0;
        IDEN(emRm);  IDEN(emIp);
        IDEN(eemm);
        #if NFLAVOR == 3
        IDEN(mtRm);  IDEN(mtIp);
        IDEN(teRm);  IDEN(teIp);
        IDEN(mmtt);
        #endif
#undef IDEN
        // Many temp variable for reduction as OpenACC doesn't support reduction into array.
        #define RV(x) x##0,x##1,x##2,x##3
        #if NFLAVOR == 3
          #pragma acc loop reduction(+:RV(emRm),RV(emIp),RV(eemm),RV(mtRm),RV(mtIp),RV(teRm),RV(teIp),RV(mmtt) )
        #elif NFLAVOR == 2
          #pragma acc loop reduction(+:RV(emRm),RV(emIp),RV(eemm))
        #endif
        #undef RV
        #pragma omp _SIMD_
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            //const real v4[] = {1,vx[v],vy[v],vz[v]};
            {
                emRm0   += vw[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
                emIp0   += vw[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
                eemm0   += vw[v]*(in->wf[ff::bee][ijkv]-in->wf[ff::bmm][ijkv]-in->wf[ff::ee][ijkv]+in->wf[ff::mm][ijkv] );
                #if NFLAVOR == 3
                mtRm0   += vw[v]*(in->wf[ff::bmtr][ijkv] - in->wf[ff::mtr][ijkv] );
                mtIp0   += vw[v]*(in->wf[ff::bmti][ijkv] + in->wf[ff::mti][ijkv] );
                teRm0   += vw[v]*(in->wf[ff::bter][ijkv] - in->wf[ff::ter][ijkv] );
                teIp0   += vw[v]*(in->wf[ff::btei][ijkv] + in->wf[ff::tei][ijkv] );
                mmtt0   += vw[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::btt][ijkv]-in->wf[ff::mm][ijkv]+in->wf[ff::tt][ijkv] );
                #endif
            }
            {
                emRm1   += vx[v]* vw[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
                emIp1   += vx[v]* vw[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
                eemm1   += vx[v]* vw[v]*(in->wf[ff::bee][ijkv]-in->wf[ff::bmm][ijkv]-in->wf[ff::ee][ijkv]+in->wf[ff::mm][ijkv] );
                #if NFLAVOR == 3
                mtRm1   += vx[v]* vw[v]*(in->wf[ff::bmtr][ijkv] - in->wf[ff::mtr][ijkv] );
                mtIp1   += vx[v]* vw[v]*(in->wf[ff::bmti][ijkv] + in->wf[ff::mti][ijkv] );
                teRm1   += vx[v]* vw[v]*(in->wf[ff::bter][ijkv] - in->wf[ff::ter][ijkv] );
                teIp1   += vx[v]* vw[v]*(in->wf[ff::btei][ijkv] + in->wf[ff::tei][ijkv] );
                mmtt1   += vx[v]* vw[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::btt][ijkv]-in->wf[ff::mm][ijkv]+in->wf[ff::tt][ijkv] );
                #endif
            }
            {
                emRm2   += vy[v]* vw[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
                emIp2   += vy[v]* vw[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
                eemm2   += vy[v]* vw[v]*(in->wf[ff::bee][ijkv]-in->wf[ff::bmm][ijkv]-in->wf[ff::ee][ijkv]+in->wf[ff::mm][ijkv] );
                #if NFLAVOR == 3
                mtRm2   += vy[v]* vw[v]*(in->wf[ff::bmtr][ijkv] - in->wf[ff::mtr][ijkv] );
                mtIp2   += vy[v]* vw[v]*(in->wf[ff::bmti][ijkv] + in->wf[ff::mti][ijkv] );
                teRm2   += vy[v]* vw[v]*(in->wf[ff::bter][ijkv] - in->wf[ff::ter][ijkv] );
                teIp2   += vy[v]* vw[v]*(in->wf[ff::btei][ijkv] + in->wf[ff::tei][ijkv] );
                mmtt2   += vy[v]* vw[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::btt][ijkv]-in->wf[ff::mm][ijkv]+in->wf[ff::tt][ijkv] );
                #endif
            }
            {
                emRm3   += vz[v]* vw[v]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
                emIp3   += vz[v]* vw[v]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
                eemm3   += vz[v]* vw[v]*(in->wf[ff::bee][ijkv]-in->wf[ff::bmm][ijkv]-in->wf[ff::ee][ijkv]+in->wf[ff::mm][ijkv] );
                #if NFLAVOR == 3
                mtRm3   += vz[v]* vw[v]*(in->wf[ff::bmtr][ijkv] - in->wf[ff::mtr][ijkv] );
                mtIp3   += vz[v]* vw[v]*(in->wf[ff::bmti][ijkv] + in->wf[ff::mti][ijkv] );
                teRm3   += vz[v]* vw[v]*(in->wf[ff::bter][ijkv] - in->wf[ff::ter][ijkv] );
                teIp3   += vz[v]* vw[v]*(in->wf[ff::btei][ijkv] + in->wf[ff::tei][ijkv] );
                mmtt3   += vz[v]* vw[v]*(in->wf[ff::bmm][ijkv]-in->wf[ff::btt][ijkv]-in->wf[ff::mm][ijkv]+in->wf[ff::tt][ijkv] );
                #endif
            }
            // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
        }

        //
        // Interaction and vacuum parts.
        //
        #pragma acc loop
        #pragma omp _SIMD_
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            // shorthand pointers
            const real ee   = in->wf[ff::ee]  [ijkv];
            const real mm   = in->wf[ff::mm]  [ijkv];
            const real emr  = in->wf[ff::emr] [ijkv];
            const real emi  = in->wf[ff::emi] [ijkv];
            const real bee  = in->wf[ff::bee] [ijkv];
            const real bmm  = in->wf[ff::bmm] [ijkv];
            const real bemr = in->wf[ff::bemr][ijkv];
            const real bemi = in->wf[ff::bemi][ijkv];
            #if NFLAVOR == 3
            const real tt   = in->wf[ff::tt]  [ijkv];
            const real mtr  = in->wf[ff::mtr] [ijkv];
            const real mti  = in->wf[ff::mti] [ijkv];
            const real ter  = in->wf[ff::ter] [ijkv];
            const real tei  = in->wf[ff::tei] [ijkv];
            const real btt  = in->wf[ff::btt] [ijkv];
            const real bmtr = in->wf[ff::bmtr][ijkv];
            const real bmti = in->wf[ff::bmti][ijkv];
            const real bter = in->wf[ff::bter][ijkv];
            const real btei = in->wf[ff::btei][ijkv];
            #endif

#define INP(x) ( x##0 - x##1 * vx[v] - x##2 * vy[v] - x##3 * vz[v])
//#define INP(x) ( x##0 - x##3 * vz[v])
#if NFLAVOR == 2
        auto iemRm = INP(emRm);
        auto iemIp = INP(emIp);
        auto ieemm = INP(eemm);

        real Iee   = 2*mu*( emi*iemRm + emr*iemIp );
        real Imm   = -Iee;
        real Iemr  =   mu*( (mm-ee)*iemIp - emi*ieemm);
        real Iemi  =   mu*( (mm-ee)*iemRm + emr*ieemm);
        real Ibee  = 2*mu*(-bemi*iemRm + bemr*iemIp);
        real Ibmm  = -Ibee;
        real Ibemr =   mu*((bmm-bee)*iemIp + bemi*ieemm );
        real Ibemi =   mu*((bee-bmm)*iemRm - bemr*ieemm );

        // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
        out->wf[ff::ee]  [ijkv] =  Iee   - pmo* 2*st*emi ;
        out->wf[ff::mm]  [ijkv] = -Iee   + pmo* 2*st*emi ;
        out->wf[ff::emr] [ijkv] =  Iemr  - pmo* 2*ct*emi ;
        out->wf[ff::emi] [ijkv] =  Iemi  + pmo*(2*ct*emr  + st*( ee - mm ) );
        out->wf[ff::bee] [ijkv] =  Ibee  - pmo* 2*st*bemi;
        out->wf[ff::bmm] [ijkv] = -Ibee  + pmo* 2*st*bemi;
        out->wf[ff::bemr][ijkv] =  Ibemr - pmo* 2*ct*bemi;
        out->wf[ff::bemi][ijkv] =  Ibemi + pmo*(2*ct*bemr + st*( bee - bmm ) );

#elif NFLAVOR == 3
        auto iemRm = INP(emRm);
        auto iemIp = INP(emIp);
        auto ieemm = INP(eemm);
        auto immtt = INP(mmtt);
        auto ittee = - ieemm - immtt;

        auto imtRm = INP(mtRm);
        auto imtIp = INP(mtIp);
        auto iteRm = INP(teRm);
        auto iteIp = INP(teIp);

        real Iee  = 2*mu*( emi*iemRm + emr*iemIp - iteIp*ter - iteRm*tei);
        real Imm  = 2*mu*(-emi*iemRm - emr*iemIp + imtIp*mtr + imtRm*mti);
        real Itt  = 2*mu*(-imtIp*mtr - imtRm*mti + iteIp*ter + iteRm*tei);
        real Iemr =  -mu*( (ee-mm)*iemIp + emi*ieemm - imtIp*ter + imtRm*tei + iteIp*mtr - iteRm*mti);
        real Iemi =   mu*( (mm-ee)*iemRm + emr*ieemm - imtIp*tei - imtRm*ter + iteIp*mti + iteRm*mtr);
        real Imtr =  -mu*( (mm-tt)*imtIp + emi*iteRm - emr*iteIp + iemIp*ter - iemRm*tei + immtt*mti);
        real Imti =   mu*(-(mm-tt)*imtRm - emi*iteIp - emr*iteRm + iemIp*tei + iemRm*ter + immtt*mtr);
        real Iter =  -mu*( (tt-ee)*iteIp - emi*imtRm + emr*imtIp - iemIp*mtr + iemRm*mti + ittee*tei);
        real Itei =   mu*(-(tt-ee)*iteRm + emi*imtIp + emr*imtRm - iemIp*mti - iemRm*mtr + ittee*ter);
        real Ibee  = 2*mu*(-bemi*iemRm + bemr*iemIp + btei*iteRm - bter*iteIp);
        real Ibmm  = 2*mu*( bemi*iemRm - bemr*iemIp - bmti*imtRm + bmtr*imtIp);
        real Ibtt  = 2*mu*( bmti*imtRm - bmtr*imtIp - btei*iteRm + bter*iteIp);
        real Ibemr =  -mu*((bee-bmm)*iemIp - bemi*ieemm + bmti*iteRm + bmtr*iteIp - btei*imtRm - bter*imtIp);
        real Ibemi =   mu*((bee-bmm)*iemRm - bemr*ieemm + bmti*iteIp - bmtr*iteRm - btei*imtIp + bter*imtRm);
        real Ibmtr =  -mu*((bmm-btt)*imtIp - bemi*iteRm - bemr*iteIp - bmti*immtt + btei*iemRm + bter*iemIp);
        real Ibmti =   mu*((bmm-btt)*imtRm - bemi*iteIp + bemr*iteRm - bmtr*immtt + btei*iemIp - bter*iemRm);
        real Ibter =  -mu*((btt-bee)*iteIp + bemi*imtRm + bemr*imtIp - bmti*iemRm - bmtr*iemIp - btei*ittee);
        real Ibtei =   mu*((btt-bee)*iteRm + bemi*imtIp - bemr*imtRm - bmti*iemIp + bmtr*iemRm - bter*ittee);

        out->wf[ff::ee  ][ijkv] = Iee   - pmo * 2*(emi*hemr + emr*hemi - htei*ter + hter*tei);
        out->wf[ff::mm  ][ijkv] = Imm   + pmo * 2*(emi*hemr - emr*hemi + hmti*mtr - hmtr*mti);
        out->wf[ff::emr ][ijkv] = Iemr  + pmo*( (mm-ee)*hemi + emi*(hee - hmm) + hmti*ter + hmtr*tei - htei*mtr - hter*mti );
        out->wf[ff::emi ][ijkv] = Iemi  + pmo*( (ee-mm)*hemr - emr*(hee - hmm) - hmti*tei + hmtr*ter + htei*mti - hter*mtr );
        out->wf[ff::bee ][ijkv] = Ibee  - pmo * 2*(bemi*hemr + bemr*hemi - htei*bter + hter*btei);
        out->wf[ff::bmm ][ijkv] = Ibmm  + pmo * 2*(bemi*hemr - bemr*hemi + hmti*bmtr - hmtr*bmti);
        out->wf[ff::bemr][ijkv] = Ibemr + pmo*( (bmm-bee)*hemi + bemi*(hee - hmm) + hmti*bter + hmtr*btei - htei*bmtr - hter*bmti );
        out->wf[ff::bemi][ijkv] = Ibemi + pmo*( (bee-bmm)*hemr - bemr*(hee - hmm) - hmti*btei + hmtr*bter + htei*bmti - hter*bmtr );
        out->wf[ff::tt  ][ijkv] = Itt   - pmo* 2*(hmti*mtr + hmtr*mti + htei*ter - hter*tei);;
        out->wf[ff::mtr ][ijkv] = Imtr  + pmo* ( (tt-mm)*hmti + emr*htei + emi*hter - hemi*ter - hemr*tei + mti*(hmm - htt) );
        out->wf[ff::mti ][ijkv] = Imti  + pmo* ( (mm-tt)*hmtr + emr*hter - emi*htei + hemi*tei - hemr*ter - mtr*(hmm - htt) );
        out->wf[ff::ter ][ijkv] = Iter  + pmo* ( (ee-tt)*htei - emr*hmti - emi*hmtr + hemi*mtr + hemr*mti - tei*(hee - htt) );
        out->wf[ff::tei ][ijkv] = Itei  + pmo* ( (tt-ee)*hter - emr*hmtr + emi*hmti - hemi*mti + hemr*mtr + ter*(hee - htt) );
        out->wf[ff::btt ][ijkv] = Ibtt  - pmo* 2*(hmti*mtr + hmtr*mti + htei*ter - hter*tei);;
        out->wf[ff::bmtr][ijkv] = Ibmtr + pmo* ( (btt-bmm)*hmti + bemr*htei + bemi*hter - hemr*btei - hemi*bter + bmti*(hmm - htt) );
        out->wf[ff::bmti][ijkv] = Ibmti + pmo* ( (bmm-btt)*hmtr + bemr*hter - bemi*htei - hemr*bter + hemi*btei - bmtr*(hmm - htt) );
        out->wf[ff::bter][ijkv] = Ibter + pmo* ( (bee-btt)*htei - bemr*hmti - bemi*hmtr + hemi*bmtr + hemr*bmti - btei*(hee - htt) );
        out->wf[ff::btei][ijkv] = Ibtei + pmo* ( (btt-bee)*hter - bemr*hmtr + bemi*hmti - hemi*bmti + hemr*bmtr + bter*(hee - htt) );

#endif  // N_FLAVOR_3
#undef INP
        }
    }
#ifdef PROFILE
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* RESTRICT v0, const FieldVar * RESTRICT v1, const real a, const FieldVar * RESTRICT v2) {
#ifdef PROFILE
    nvtxRangePush("Vec1");
#endif

    for(int f=0;f<nvar;++f)   // field-loop in the outer hoping to make memory access continuous ( but not effective )
    PARFORALL(i,j,k,v) {
        auto ijkv = idx(i,j,k,v);
        v0->wf[f][ijkv] = v1->wf[f][ijkv] + a * v2->wf[f][ijkv];
    }
#ifdef PROFILE
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* RESTRICT v0, const FieldVar * RESTRICT v1, const real a, const FieldVar * RESTRICT v2, const FieldVar * RESTRICT v3) {
#ifdef PROFILE
    nvtxRangePush("Vec2");
#endif

    for(int f=0;f<nvar;++f)   // field-loop in the outer hoping to make memory access continuous ( but not effective )
    PARFORALL(i,j,k,v) {
        auto ijkv = idx(i,j,k,v);
        v0->wf[f][ijkv] = v1->wf[f][ijkv] + a * (v2->wf[f][ijkv] + v3->wf[f][ijkv]);
    }
#ifdef PROFILE
    nvtxRangePop();
#endif
}

void NuOsc::step_rk4() {
#ifdef PROFILE
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
    iter++;
#ifdef PROFILE
    nvtxRangePop();
#endif
}

#ifdef SCHEME_WENO7
void NuOsc::get_flux(Flux * RESTRICT out_flux, const real *in_field, const int stride, const int xdelta = 0, const int ydelta = 0, const int zdelta = 0)
{
#ifdef PROFILE
    nvtxRangePush("Flux");
#endif
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

    #pragma acc parallel loop independent collapse(4)
    #pragma omp parallel for collapse(3)
    for (int xid = -xdelta; xid < nx[0] + xdelta; xid++)
    for (int yid = -ydelta; yid < nx[1] + ydelta; yid++)
    for (int zid = -zdelta; zid < nx[2] + zdelta; zid++) {
    #pragma omp _SIMD_
    for (int bin = 0; bin < nv; bin++)    {
                    auto ijkv = idx(xid, yid, zid, bin);
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
#ifdef PROFILE
    nvtxRangePop();
#endif
}
#endif
