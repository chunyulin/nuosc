#include "nuosc_class.h"

void NuOsc::eval_conserved(const FieldVar* __restrict v0) {
#ifdef NVTX
    nvtxRangePush("eval_conserved");
#endif

    PARFORALL(i,j,v)   {
        uint ijv = grid.idx(i,j,v);
        real iG  = 1.0 / G0[ijv];
        real iGb = 1.0 / G0b[ijv];
        P1 [ijv] =   2.0*v0->ex_re[ijv] * iG;
        P2 [ijv] = - 2.0*v0->ex_im[ijv] * iG;
        P3 [ijv] = (v0->ee[ijv] - v0->xx[ijv])*iG;
        P1b[ijv] =  2.0*v0->bex_re[ijv] * iGb;
        P2b[ijv] =  2.0*v0->bex_im[ijv] * iGb;
        P3b[ijv] = (v0->bee[ijv] - v0->bxx[ijv]) * iGb;

        dN [ijv] = (v0->ee [ijv] + v0->xx [ijv]);
        dN [ijv] = (dN [ijv] - G0[ijv])/dN [ijv];   // relative difference of (ee+xx)
        dNb[ijv] = (v0->bee[ijv] + v0->bxx[ijv]);
        dNb[ijv] = (dNb [ijv] - G0b[ijv])/dNb [ijv] ;
        //dN [ijv] = ( (v0->ee [ijv] + v0->xx [ijv]) - G0[ijv])  / (v0->ee [ijv] + v0->xx [ijv]) ;   // relative difference of (ee+xx)
        //dNb[ijv] = ( (v0->bee[ijv] + v0->bxx[ijv]) - G0b[ijv]) / (v0->bee[ijv] + v0->bxx[ijv]) ;

        dP [ijv] = std::abs( 1.0 - std::sqrt(P1 [ijv]*P1 [ijv]+P2 [ijv]*P2 [ijv]+P3 [ijv]*P3 [ijv]) );
        dPb[ijv] = std::abs( 1.0 - std::sqrt(P1b[ijv]*P1b[ijv]+P2b[ijv]*P2b[ijv]+P3b[ijv]*P3b[ijv]) );
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::analysis() {
#ifdef NVTX
    nvtxRangePush("analysis");
#endif

    eval_conserved(v_stat);

    real maxdP = 0.0;
    real maxdN = 0.0;
    real avgP  = 0.0, avgPb = 0.0;
    real surv  = 0.0, survb  = 0.0;
    real s0  = 0.0, s1  = 0.0;
    real nor  = 0.0, norb = 0.0;
    real aM01 = 0.0, aM02 = 0.0, aM03 = 0.0;
    //real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;
#ifdef ADV_TEST
    real I1=0., I2=0.;
#endif
    // integral over (vz,z): assume dz=dy=const.
#pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP,maxdN) collapse(COLLAPSE_LOOP)
#pragma acc parallel loop reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP,maxdN) collapse(COLLAPSE_LOOP)
    FORALL(i,j,v)  {
        int ijv = grid.idx(i,j,v);

#ifdef ADV_TEST
        I1 += grid.vw[v]* v_stat->ee [ijv];
        I2 += grid.vw[v]* v_stat->ee [ijv]* v_stat->ee [ijv];
#endif

        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        maxdP = std::max( maxdP, std::max(dP[ijv], dPb[ijv]));
        //maxdN = std::max( std::max(maxdN,dN[ijv]), dNb[ijv]);  // What's this?

        surv  += grid.vw[v]* v_stat->ee [ijv];
        survb += grid.vw[v]* v_stat->bee[ijv];

        avgP  += grid.vw[v] * G0 [ijv] * std::abs( 1.0 - std::sqrt(P1 [ijv]*P1 [ijv]+P2 [ijv]*P2 [ijv]+P3 [ijv]*P3 [ijv]) );
        avgPb += grid.vw[v] * G0b[ijv] * std::abs( 1.0 - std::sqrt(P1b[ijv]*P1b[ijv]+P2b[ijv]*P2b[ijv]+P3b[ijv]*P3b[ijv]) );

        // M0
        aM01 += grid.vw[v]* ( v_stat->ex_re[ijv] - v_stat->bex_re[ijv]);                                 // = P1[ijv]*G0[ijv] - P1b[ijv]*G0b[ijv];
        aM02 += grid.vw[v]* (-v_stat->ex_im[ijv] - v_stat->bex_im[ijv]);                                 // = P2[ijv]*G0[ijv] - P2b[ijv]*G0b[ijv];
        aM03 += grid.vw[v]* 0.5*(v_stat->ee[ijv] - v_stat->xx[ijv] - v_stat->bee[ijv] + v_stat->bxx[ijv]); // = P3[ijv]*G0[ijv] - P3b[ijv]*G0b[ijv], which is also the net e-x lepton number;

        // M1
        //aM11 += grid.vw[v]* grid.vz[i]* (v_stat->ex_re[ijv] - v_stat->bex_re[ijv]);
        //aM12 += grid.vw[v]* grid.vz[i]* (v_stat->ex_im[ijv] + v_stat->bex_im[ijv]);
        //aM13 += grid.vw[v]* grid.vz[i]* 0.5*(v_stat->ee[ijv] - v_stat->xx[ijv] - v_stat->bee[ijv] + v_stat->bxx[ijv]);

        nor  += grid.vw[v]* G0 [ijv];   // const: should be calculated initially
        norb += grid.vw[v]* G0b[ijv];
    }

#ifdef COSENU_MPI
    // TODO: pack multiple MPI_reduce into one.
#define REDUCE(x, op) \
    if (myrank==0)  MPI_Reduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, op, 0, grid.CartCOMM); \
    else            MPI_Reduce(&x,           &x, 1, MPI_DOUBLE, op, 0, grid.CartCOMM);
    //MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, op, grid.CartCOMM);

    REDUCE(surv,MPI_SUM);   REDUCE(survb,MPI_SUM);
    REDUCE(avgP,MPI_SUM);   REDUCE(avgPb,MPI_SUM);
    REDUCE(nor, MPI_SUM);   REDUCE(norb, MPI_SUM);
    REDUCE(aM01,MPI_SUM);   REDUCE(aM02, MPI_SUM);    REDUCE(aM03,MPI_SUM);
    REDUCE(maxdP,MPI_MAX);  /// !!!
#undef REDUCE(x,op)
#endif


    if (!myrank) {

        surv  /= nor;
        survb /= norb;
        avgP  /= nor;   /// !!!
        avgPb /= norb;  /// !!!
        real aM0    = std::sqrt(aM01*aM01+aM02*aM02+aM03*aM03) * ds_L;
        real ELNe  = std::abs(n_nue0[0]*(1.0-surv) - n_nue0[1]*(1.0-survb)) / (n_nue0[0]+n_nue0[1]); /// !!!
        //real ELNe2 = std::abs(1.0*(1.0-surv) - 0.9*(1-survb)) / (1.9);
        real Lex = aM03 * ds_L;

        printf("T= %15f ", phy_time);
#ifdef ADV_TEST
        printf(" I1= %5.4e I2= %5.4e\n", I1/nor, I2/nor);
#else
        //printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e lN= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, aM03);
        printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e ELNe= %g Lex= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, ELNe, Lex);
#endif
        anafile << phy_time << std::setprecision(13) << " " << maxdP << " " 
            << surv << " " << survb << " " 
            << avgP << " " << avgPb << " " 
            << aM0 << " " << Lex  << " " << ELNe << " " << endl;

        assert(maxdP <10 );

    }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::renormalize(const FieldVar* __restrict v0) {

    PARFORALL(i,j,v)   {
        int ijv = grid.idx(i,j,v);
        real iG = 1.0 / G0[ijv];
        real iGb = 1.0 / G0b[ijv];
        real P1  =   v0->ex_re[ijv] * iG;
        real P2  = - v0->ex_im[ijv] * iG;
        real P3  = (v0->ee[ijv] - v0->xx[ijv])*iG;
        real P1b = v0->bex_re[ijv] * iGb;
        real P2b = v0->bex_im[ijv] * iGb;
        real P3b = (v0->bee[ijv] - v0->bxx[ijv]) * iGb;
        real iP   = 1.0/std::sqrt(P1*P1+P2*P2+P3*P3);
        real iPb  = 1.0/std::sqrt(P1b*P1b+P2b*P2b+P3b*P3b);
        real tmp  = iP *(P3) *G0 [ijv];
        real tmpb = iPb*(P3b)*G0b[ijv];

        v0->ee    [ijv]  = G0[ijv]+tmp;
        v0->xx    [ijv]  = G0[ijv]-tmp;
        v0->ex_re [ijv] *= iP;
        v0->ex_im [ijv] *= iP;
        v0->bee   [ijv]  = G0b[ijv]+tmpb;
        v0->bxx   [ijv]  = G0b[ijv]-tmpb;
        v0->bex_re[ijv] *= iPb;
        v0->bex_im[ijv] *= iPb;
    }
}

