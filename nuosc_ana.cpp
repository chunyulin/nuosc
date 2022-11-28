#include "nuosc_class.h"

void NuOsc::eval_conserved(const FieldVar* __restrict v0) {
#ifdef NVTX
    nvtxRangePush("Eval conserved");
#endif

    PARFORALL(i,j,k,v)   {
        uint ijkv = grid.idx(i,j,k,v);
        real iG  = 1.0 / G0[ijkv];
        real iGb = 1.0 / G0b[ijkv];
        P1 [ijkv] =   2.0*v0->ex_re[ijkv] * iG;
        P2 [ijkv] = - 2.0*v0->ex_im[ijkv] * iG;
        P3 [ijkv] = (v0->ee[ijkv] - v0->xx[ijkv])*iG;
        P1b[ijkv] =  2.0*v0->bex_re[ijkv] * iGb;
        P2b[ijkv] =  2.0*v0->bex_im[ijkv] * iGb;
        P3b[ijkv] = (v0->bee[ijkv] - v0->bxx[ijkv]) * iGb;

        dN [ijkv] = (v0->ee [ijkv] + v0->xx [ijkv]);
        dN [ijkv] = (dN [ijkv] - G0[ijkv])/dN [ijkv];   // relative difference of (ee+xx)
        dNb[ijkv] = (v0->bee[ijkv] + v0->bxx[ijkv]);
        dNb[ijkv] = (dNb [ijkv] - G0b[ijkv])/dNb [ijkv] ;
        //dN [ijkv] = ( (v0->ee [ijkv] + v0->xx [ijkv]) - G0[ijkv])  / (v0->ee [ijkv] + v0->xx [ijkv]) ;   // relative difference of (ee+xx)
        //dNb[ijkv] = ( (v0->bee[ijkv] + v0->bxx[ijkv]) - G0b[ijkv]) / (v0->bee[ijkv] + v0->bxx[ijkv]) ;

        dP [ijkv] = std::abs( 1.0 - std::sqrt(P1 [ijkv]*P1 [ijkv]+P2 [ijkv]*P2 [ijkv]+P3 [ijkv]*P3 [ijkv]) );
        dPb[ijkv] = std::abs( 1.0 - std::sqrt(P1b[ijkv]*P1b[ijkv]+P2b[ijkv]*P2b[ijkv]+P3b[ijkv]*P3b[ijkv]) );
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::analysis() {
#ifdef NVTX
    nvtxRangePush("Analysis");
#endif

    eval_conserved(v_stat);

    real maxdP = 0.0;
    //real maxdN = 0.0;
    real avgP  = 0.0, avgPb = 0.0;
    real surv  = 0.0, survb  = 0.0;
    real nor  = 0.0, norb = 0.0;
    real aM01 = 0.0, aM02 = 0.0, aM03 = 0.0;
    //real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;
#ifdef ADV_TEST
    real I1=0., I2=0.;
    #pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb,I1,I2) reduction(max:maxdP) collapse(COLLAPSE_LOOP)
    #pragma acc parallel loop reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb,I1,I2) reduction(max:maxdP) collapse(COLLAPSE_LOOP)
#else
    #pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP) collapse(COLLAPSE_LOOP)
    #pragma acc parallel loop reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP) collapse(COLLAPSE_LOOP)
#endif
    FORALL(i,j,k,v)  {
        int ijkv = grid.idx(i,j,k,v);

#ifdef ADV_TEST
        I1 += grid.vw[v]* v_stat->ee [ijkv];
        I2 += grid.vw[v]* v_stat->ee [ijkv]* v_stat->ee [ijkv];
#endif

        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        maxdP = std::max( maxdP, std::max(dP[ijkv], dPb[ijkv]));
        //maxdN = std::max( std::max(maxdN,dN[ijkv]), dNb[ijkv]);  // What's this?

        surv  += grid.vw[v]* v_stat->ee [ijkv];
        survb += grid.vw[v]* v_stat->bee[ijkv];

        avgP  += grid.vw[v] * G0 [ijkv] * std::abs( 1.0 - std::sqrt(P1 [ijkv]*P1 [ijkv]+P2 [ijkv]*P2 [ijkv]+P3 [ijkv]*P3 [ijkv]) );
        avgPb += grid.vw[v] * G0b[ijkv] * std::abs( 1.0 - std::sqrt(P1b[ijkv]*P1b[ijkv]+P2b[ijkv]*P2b[ijkv]+P3b[ijkv]*P3b[ijkv]) );

        // M0
        aM01 += grid.vw[v]* ( v_stat->ex_re[ijkv] - v_stat->bex_re[ijkv]);                                 // = P1[ijkv]*G0[ijkv] - P1b[ijkv]*G0b[ijkv];
        aM02 += grid.vw[v]* (-v_stat->ex_im[ijkv] - v_stat->bex_im[ijkv]);                                 // = P2[ijkv]*G0[ijkv] - P2b[ijkv]*G0b[ijkv];
        aM03 += grid.vw[v]* 0.5*(v_stat->ee[ijkv] - v_stat->xx[ijkv] - v_stat->bee[ijkv] + v_stat->bxx[ijkv]); // = P3[ijkv]*G0[ijkv] - P3b[ijkv]*G0b[ijkv], which is also the net e-x lepton number;

        // M1
        //aM11 += grid.vw[v]* grid.vz[i]* (v_stat->ex_re[ijkv] - v_stat->bex_re[ijkv]);
        //aM12 += grid.vw[v]* grid.vz[i]* (v_stat->ex_im[ijkv] + v_stat->bex_im[ijkv]);
        //aM13 += grid.vw[v]* grid.vz[i]* 0.5*(v_stat->ee[ijkv] - v_stat->xx[ijkv] - v_stat->bee[ijkv] + v_stat->bxx[ijkv]);

        nor  += grid.vw[v]* G0 [ijkv];   // const: should be calculated initially
        norb += grid.vw[v]* G0b[ijkv];
    }

#ifdef COSENU_MPI
    // TODO: pack multiple MPI_reduce into one.
#define REDUCE(x, op) \
    if (!myrank)  MPI_Reduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, op, 0, grid.CartCOMM); \
    else          MPI_Reduce(&x,           &x, 1, MPI_DOUBLE, op, 0, grid.CartCOMM);

    //MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, op, grid.CartCOMM);
    REDUCE(surv,MPI_SUM);   REDUCE(survb,MPI_SUM);
    REDUCE(avgP,MPI_SUM);   REDUCE(avgPb,MPI_SUM);
    REDUCE(nor, MPI_SUM);   REDUCE(norb, MPI_SUM);
    REDUCE(aM01,MPI_SUM);   REDUCE(aM02, MPI_SUM);    REDUCE(aM03,MPI_SUM);
    REDUCE(maxdP,MPI_MAX);
#undef REDUCE
#endif


    if (!myrank) {

        surv  /= nor;   avgP  /= nor;
        survb /= norb;  avgPb /= norb;

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

    PARFORALL(i,j,k,v)   {
        int ijkv = grid.idx(i,j,k,v);
        real iG = 1.0 / G0[ijkv];
        real iGb = 1.0 / G0b[ijkv];
        real P1  =   v0->ex_re[ijkv] * iG;
        real P2  = - v0->ex_im[ijkv] * iG;
        real P3  = (v0->ee[ijkv] - v0->xx[ijkv])*iG;
        real P1b = v0->bex_re[ijkv] * iGb;
        real P2b = v0->bex_im[ijkv] * iGb;
        real P3b = (v0->bee[ijkv] - v0->bxx[ijkv]) * iGb;
        real iP   = 1.0/std::sqrt(P1*P1+P2*P2+P3*P3);
        real iPb  = 1.0/std::sqrt(P1b*P1b+P2b*P2b+P3b*P3b);
        real tmp  = iP *(P3) *G0 [ijkv];
        real tmpb = iPb*(P3b)*G0b[ijkv];

        v0->ee    [ijkv]  = G0[ijkv]+tmp;
        v0->xx    [ijkv]  = G0[ijkv]-tmp;
        v0->ex_re [ijkv] *= iP;
        v0->ex_im [ijkv] *= iP;
        v0->bee   [ijkv]  = G0b[ijkv]+tmpb;
        v0->bxx   [ijkv]  = G0b[ijkv]-tmpb;
        v0->bex_re[ijkv] *= iPb;
        v0->bex_im[ijkv] *= iPb;
    }
}

