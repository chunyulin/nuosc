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
    calRHS_wo_bdry(out, in);  // overlap with non-blocking sync
#ifdef PROFILING
if (!myrank) printf("  [%.4f] calRHS_wo_bdry done.\n", utils::msecs_since());
#endif
    waitall();
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Waitall.\n", utils::msecs_since());
#endif
    unpack_buffer(in);
#ifdef PROFILING
if (!myrank) printf("  [%.4f] Unpack.\n", utils::msecs_since());
#endif
    calRHS_with_bdry(out, in);
#ifdef PROFILING
if (!myrank) printf("  [%.4f] calRHS_with_bdry done.\n", utils::msecs_since());
#endif
    #else
    updatePeriodicBoundary(in);
    calRHS_wo_bdry(out, in);  // overlap with non-blocking sync
    calRHS_with_bdry(out, in);
    #endif
}

void NuOsc::calRHS_with_bdry(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("calRHS_with_bdry");
#endif
    // for abbreviation
    const int nzv  = nv* (nx[2] + 2*gx[2]);
    const int nyzv = nzv*(nx[1] + 2*gx[1]);

    for (int f=0;f<nvar; ++f) {

#ifdef WENO7
        // adv-x
        #pragma acc parallel loop independent collapse(4)
        #pragma omp parallel for simd collapse(4)
        FORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            get_flux(flux, in->wf[f], nyzv, 0, 0, 0);
            int s = sgn(vx[v]);
            out->wf[f][ijkv] += -0.5*vx[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nyzv]) + std::abs(1-s)*(flux->h2l[ijkv+nyzv]-flux->h2l[ijkv]));
        }
        // adv-y
        #pragma acc parallel loop independent collapse(4)
        #pragma omp parallel for simd collapse(4)
        FORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            get_flux(flux, in->wf[f], nzv, 0, 0, 0);
            int s = sgn(vy[v]);
            out->wf[f][ijkv] += -0.5*vy[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nzv]) + std::abs(1-s)*(flux->h2l[ijkv+nzv]-flux->h2l[ijkv]));
        }
        // adv-z
        #pragma acc parallel loop independent collapse(4)
        #pragma omp parallel for simd collapse(4)
        FORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            get_flux(flux, in->wf[f], nv, 0, 0, 0);
            int s = sgn(vz[v]);
            out->wf[f][ijkv] += -0.5*vz[v]/dx * (std::abs(1+s) * (flux->l2h[ijkv]-flux->l2h[ijkv-nv]) + std::abs(1-s)*(flux->h2l[ijkv+nv]-flux->h2l[ijkv]));
        }
#else
        #pragma acc parallel loop independent collapse(4)
        #pragma omp parallel for simd collapse(4)
        FORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);
            const real *ff   = &(in->wf[f][ijkv]);
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
    
            // prepare KO operator
            #ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps = -ko/dx/64.0;
            #define KO_FD(x) ko_eps*( \
               ( x[-3*nv] + x[3*nv] - 6*(x[-2*nv] +x[2*nv])  + 15*(x[-nv] + x[nv]) - 20*x[0] ) + \
               ( x[-3*nzv]   + x[3*nzv]   - 6*(x[-2*nzv]     +x[2*nzv])      + 15*(x[-nzv]     + x[nzv])     - 20*x[0] ) + \
               ( x[-3*nyzv]  + x[3*nyzv]  - 6*(x[-2*nyzv]    +x[2*nyzv])     + 15*(x[-nyzv]    + x[nyzv])    - 20*x[0] ) )
            #else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps = -ko/dx/16.0;
            #define KO_FD(x) ko_eps*( \
               ( x[-2*nv] + x[2*nv] - 4*(x[-nv] + x[nv]) + 6*x[0] ) + \
               ( x[-2*nzv]  + x[2*nzv]     - 4*(x[-nzv]     + x[nzv])     + 6*x[0] ) + \
               ( x[-2*nyzv] + x[2*nyzv]    - 4*(x[-nyzv]    + x[nyzv])    + 6*x[0] ) )
            #endif
            out->wf[f][ijkv] += ADV_FD(ff) + KO_FD(ff);
        } // end for xyzv.
#endif  // end if WENO7
    } // end for fields.

#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::calRHS_wo_bdry(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush("calRHS_wo_bdry");
#endif
    #pragma acc parallel loop independent collapse(3)
    #pragma ivdeps
    #pragma omp parallel for collapse(3)
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<nx[2]; ++k) {
        //
        // common integral factors over vz'
        //
        std::array<real,4> emRm{{0,0,0,0}};
        std::array<real,4> emIp{{0,0,0,0}};
        std::array<real,4> eemm{{0,0,0,0}};
        #if NFLAVOR == 3
        std::array<real,4> mtRm{{0,0,0,0}};
        std::array<real,4> mtIp{{0,0,0,0}};
        std::array<real,4> teRm{{0,0,0,0}};
        std::array<real,4> teIp{{0,0,0,0}};
        std::array<real,4> mmtt{{0,0,0,0}};
        #endif

#define IDEN(x) real &x##0 = x[0]; real &x##1 = x[1]; real &x##2 = x[2]; real &x##3 = x[3];
        IDEN(emRm);  IDEN(emIp);
        IDEN(eemm);
        #if NFLAVOR == 3
        IDEN(mtRm);  IDEN(mtIp);
        IDEN(teRm);  IDEN(teIp);
        IDEN(mmtt);
        #endif
#undef IDEN

        #define RV(x) x##0,x##1,x##2,x##3
        #if NFLAVOR == 3
          #pragma acc loop reduction(+:RV(emRm),RV(emIp),RV(eemm),RV(mtRm),RV(mtIp),RV(teRm),RV(teIp),RV(mmtt) )
        #elif NFLAVOR == 2
          #pragma acc loop reduction(+:RV(emRm),RV(emIp),RV(eemm))
        #endif
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            const std::array<real,4> v4{1,vx[k],vy[k],vz[k]};
            #pragma omp simd
            for(int s=0;s<4;++s) {
                emRm[s]   += v4[s]* vw[k]*(in->wf[ff::bemr][ijkv] - in->wf[ff::emr][ijkv] );
                emIp[s]   += v4[s]* vw[k]*(in->wf[ff::bemi][ijkv] + in->wf[ff::emi][ijkv] );
                eemm[s]   += v4[s]* vw[k]*(in->wf[ff::bee][ijkv]-in->wf[ff::bmm][ijkv]-in->wf[ff::ee][ijkv]+in->wf[ff::mm][ijkv] );
                #if NFLAVOR == 3
                mtRm[s]   += v4[s]* vw[k]*(in->wf[ff::bmtr][ijkv] - in->wf[ff::mtr][ijkv] );
                mtIp[s]   += v4[s]* vw[k]*(in->wf[ff::bmti][ijkv] + in->wf[ff::mti][ijkv] );
                teRm[s]   += v4[s]* vw[k]*(in->wf[ff::bter][ijkv] - in->wf[ff::ter][ijkv] );
                teIp[s]   += v4[s]* vw[k]*(in->wf[ff::btei][ijkv] + in->wf[ff::tei][ijkv] );
                mmtt[s]   += v4[s]* vw[k]*(in->wf[ff::bmm][ijkv]-in->wf[ff::btt][ijkv]-in->wf[ff::mm][ijkv]+in->wf[ff::tt][ijkv] );
                #endif
            }
            // integral over vx and vy is zero for axi-symm case.  ( VY TO BE CHECKED...)
        }

        //
        // Interaction and vacuum parts.
        //
        #pragma acc loop
        #pragma omp simd
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);
            // shorthand pointers
            const real *ee   = &(in->wf[ff::ee]  [ijkv]);
            const real *mm   = &(in->wf[ff::mm]  [ijkv]);
            const real *emr  = &(in->wf[ff::emr] [ijkv]);
            const real *emi  = &(in->wf[ff::emi] [ijkv]);
            const real *bee  = &(in->wf[ff::bee] [ijkv]);
            const real *bmm  = &(in->wf[ff::bmm] [ijkv]);
            const real *bemr = &(in->wf[ff::bemr][ijkv]);
            const real *bemi = &(in->wf[ff::bemi][ijkv]);
            #if NFLAVOR == 3
            const real *tt   = &(in->wf[ff::tt]  [ijkv]);
            const real *mtr  = &(in->wf[ff::mtr] [ijkv]);
            const real *mti  = &(in->wf[ff::mti] [ijkv]);
            const real *ter  = &(in->wf[ff::ter] [ijkv]);
            const real *tei  = &(in->wf[ff::tei] [ijkv]);
            const real *btt  = &(in->wf[ff::btt] [ijkv]);
            const real *bmtr = &(in->wf[ff::bmtr][ijkv]);
            const real *bmti = &(in->wf[ff::bmti][ijkv]);
            const real *bter = &(in->wf[ff::bter][ijkv]);
            const real *btei = &(in->wf[ff::btei][ijkv]);
            #endif

#define INP(x) (x[0]-x[1]*vx[v]-x[2]*vy[v]-x[3]*vz[v])
#if NFLAVOR == 2
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
#endif  // N_FLAVOR_3
#undef INP
        #if NFLAVOR == 3
        out->wf[ff::ee  ][ijkv] = Iee   - pmo * 2*(emi[0]*hemr + emr[0]*hemi - htei*ter[0] + hter*tei[0]);
        out->wf[ff::mm  ][ijkv] = Imm   + pmo * 2*(emi[0]*hemr - emr[0]*hemi + hmti*mtr[0] - hmtr*mti[0]);
        out->wf[ff::emr ][ijkv] = Iemr  + pmo*( (mm[0]-ee[0])*hemi + emi[0]*(hee - hmm) + hmti*ter[0] + hmtr*tei[0] - htei*mtr[0] - hter*mti[0] );
        out->wf[ff::emi ][ijkv] = Iemi  + pmo*( (ee[0]-mm[0])*hemr - emr[0]*(hee - hmm) - hmti*tei[0] + hmtr*ter[0] + htei*mti[0] - hter*mtr[0] );
        out->wf[ff::bee ][ijkv] = Ibee  - pmo * 2*(bemi[0]*hemr + bemr[0]*hemi - htei*bter[0] + hter*btei[0]);
        out->wf[ff::bmm ][ijkv] = Ibmm  + pmo * 2*(bemi[0]*hemr - bemr[0]*hemi + hmti*bmtr[0] - hmtr*bmti[0]);
        out->wf[ff::bemr][ijkv] = Ibemr + pmo*( (bmm[0]-bee[0])*hemi + bemi[0]*(hee - hmm) + hmti*bter[0] + hmtr*btei[0] - htei*bmtr[0] - hter*bmti[0] );
        out->wf[ff::bemi][ijkv] = Ibemi + pmo*( (bee[0]-bmm[0])*hemr - bemr[0]*(hee - hmm) - hmti*btei[0] + hmtr*bter[0] + htei*bmti[0] - hter*bmtr[0] );
        out->wf[ff::tt  ][ijkv] = Itt   - pmo* 2*(hmti*mtr[0] + hmtr*mti[0] + htei*ter[0] - hter*tei[0]);;
        out->wf[ff::mtr ][ijkv] = Imtr  + pmo* ( (tt[0]-mm[0])*hmti + emr[0]*htei + emi[0]*hter - hemi*ter[0] - hemr*tei[0] + mti[0]*(hmm - htt) );
        out->wf[ff::mti ][ijkv] = Imti  + pmo* ( (mm[0]-tt[0])*hmtr + emr[0]*hter - emi[0]*htei + hemi*tei[0] - hemr*ter[0] - mtr[0]*(hmm - htt) );
        out->wf[ff::ter ][ijkv] = Iter  + pmo* ( (ee[0]-tt[0])*htei - emr[0]*hmti - emi[0]*hmtr + hemi*mtr[0] + hemr*mti[0] - tei[0]*(hee - htt) );
        out->wf[ff::tei ][ijkv] = Itei  + pmo* ( (tt[0]-ee[0])*hter - emr[0]*hmtr + emi[0]*hmti - hemi*mti[0] + hemr*mtr[0] + ter[0]*(hee - htt) );
        out->wf[ff::btt ][ijkv] = Ibtt  - pmo* 2*(hmti*mtr[0] + hmtr*mti[0] + htei*ter[0] - hter*tei[0]);;
        out->wf[ff::bmtr][ijkv] = Ibmtr + pmo* ( (btt[0]-bmm[0])*hmti + bemr[0]*htei + bemi[0]*hter - hemr*btei[0] - hemi*bter[0] + bmti[0]*(hmm - htt) );
        out->wf[ff::bmti][ijkv] = Ibmti + pmo* ( (bmm[0]-btt[0])*hmtr + bemr[0]*hter - bemi[0]*htei - hemr*bter[0] + hemi*btei[0] - bmtr[0]*(hmm - htt) );
        out->wf[ff::bter][ijkv] = Ibter + pmo* ( (bee[0]-btt[0])*htei - bemr[0]*hmti - bemi[0]*hmtr + hemi*bmtr[0] + hemr*bmti[0] - btei[0]*(hee - htt) );
        out->wf[ff::btei][ijkv] = Ibtei + pmo* ( (btt[0]-bee[0])*hter - bemr[0]*hmtr + bemi[0]*hmti - hemi*bmti[0] + hemr*bmtr[0] + bter[0]*(hee - htt) );
        #elif NFLAVOR == 2
        // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
        out->wf[ff::ee]  [ijkv] =  Iee   - pmo* 2*st*emi [0];
        out->wf[ff::mm]  [ijkv] = -Iee   + pmo* 2*st*emi [0];
        out->wf[ff::emr] [ijkv] =  Iemr  - pmo* 2*ct*emi [0];
        out->wf[ff::emi] [ijkv] =  Iemi  + pmo*(2*ct*emr [0] + st*( ee[0] - mm[0] ) );
        out->wf[ff::bee] [ijkv] =  Ibee  - pmo* 2*st*bemi[0];
        out->wf[ff::bmm] [ijkv] = -Ibee  + pmo* 2*st*bemi[0];
        out->wf[ff::bemr][ijkv] =  Ibemr - pmo* 2*ct*bemi[0];
        out->wf[ff::bemi][ijkv] =  Ibemi + pmo*(2*ct*bemr[0] + st*( bee[0] - bmm[0] ) );
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

#ifdef WENO7
void NuOsc::get_flux(Flux *out_flux, const std::vector<real> in_field, const int stride, const int xdelta = 0, const int ydelta = 0, const int zdelta = 0)
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

    #pragma acc parallel loop collapse(3)
    #pragma omp parallel for simd collapse(4)
    for (int xid = -xdelta; xid < nx[0] + xdelta; xid++)
    for (int yid = -ydelta; yid < nx[1] + ydelta; yid++)
    for (int zid = -zdelta; zid < nx[2] + zdelta; zid++)
    #pragma acc loop seq
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
    }
}
#endif
