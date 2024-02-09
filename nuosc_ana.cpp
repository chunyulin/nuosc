#include "nuosc_class.h"

void NuOsc::eval_conserved(const FieldVar* __restrict v0) {
#ifdef NVTX
    nvtxRangePush("Eval conserved");
#endif

    PARFORALL(i,j,k,v)   {
        auto ijkv = idx(i,j,k,v);
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

    // packed reduction variable for MPI send. TODO: check if these work for OpenACC 
    real rv[10] = {0};

    real t_surv  = rv[0], t_survb = rv[1];
    real t_avgP  = rv[2], t_avgPb = rv[3];
    real t_nor  =  rv[4], t_norb =  rv[5];
    real t_aM01 =  rv[6], t_aM02 =  rv[7], t_aM03 =  rv[8];
    real t_maxdP = rv[9];

#ifdef ADV_TEST
    real I1=0., I2=0.;
    #pragma omp parallel for  reduction(+:t_avgP,t_avgPb,t_aM01,t_aM02,t_aM03,t_nor,t_norb,t_surv,t_survb,I1,I2) reduction(max:t_maxdP) collapse(COLLAPSE_LOOP)
    #pragma acc parallel loop reduction(+:t_avgP,t_avgPb,t_aM01,t_aM02,t_aM03,t_nor,t_norb,t_surv,t_survb,I1,I2) reduction(max:t_maxdP) collapse(COLLAPSE_LOOP)
#else
    #pragma omp parallel for  reduction(+:t_avgP,t_avgPb,t_aM01,t_aM02,t_aM03,t_nor,t_norb,t_surv,t_survb) reduction(max:t_maxdP) collapse(COLLAPSE_LOOP)
    #pragma acc parallel loop reduction(+:t_avgP,t_avgPb,t_aM01,t_aM02,t_aM03,t_nor,t_norb,t_surv,t_survb) reduction(max:t_maxdP) collapse(COLLAPSE_LOOP)
#endif
    FORALL(i,j,k,v)  {
        int ijkv = idx(i,j,k,v);

#ifdef ADV_TEST
        I1 += vw[v]* v_stat->ee [ijkv];
        I2 += vw[v]* v_stat->ee [ijkv]* v_stat->ee [ijkv];
#endif

        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        t_maxdP = std::max( t_maxdP, std::max(dP[ijkv], dPb[ijkv]));
        //maxdN = std::max( std::max(maxdN,dN[ijkv]), dNb[ijkv]);  // What's this?

        t_surv  += vw[v]* v_stat->ee [ijkv];
        t_survb += vw[v]* v_stat->bee[ijkv];

        t_avgP  += vw[v] * G0 [ijkv] * std::abs( 1.0 - std::sqrt(P1 [ijkv]*P1 [ijkv]+P2 [ijkv]*P2 [ijkv]+P3 [ijkv]*P3 [ijkv]) );
        t_avgPb += vw[v] * G0b[ijkv] * std::abs( 1.0 - std::sqrt(P1b[ijkv]*P1b[ijkv]+P2b[ijkv]*P2b[ijkv]+P3b[ijkv]*P3b[ijkv]) );

        // M0
        t_aM01 += vw[v]* ( v_stat->ex_re[ijkv] - v_stat->bex_re[ijkv]);                                 // = P1[ijkv]*G0[ijkv] - P1b[ijkv]*G0b[ijkv];
        t_aM02 += vw[v]* (-v_stat->ex_im[ijkv] - v_stat->bex_im[ijkv]);                                 // = P2[ijkv]*G0[ijkv] - P2b[ijkv]*G0b[ijkv];
        t_aM03 += vw[v]* 0.5*(v_stat->ee[ijkv] - v_stat->xx[ijkv] - v_stat->bee[ijkv] + v_stat->bxx[ijkv]); // = P3[ijkv]*G0[ijkv] - P3b[ijkv]*G0b[ijkv], which is also the net e-x lepton number;

        // M1
        //aM11 += vw[v]* vz[i]* (v_stat->ex_re[ijkv] - v_stat->bex_re[ijkv]);
        //aM12 += vw[v]* vz[i]* (v_stat->ex_im[ijkv] + v_stat->bex_im[ijkv]);
        //aM13 += vw[v]* vz[i]* 0.5*(v_stat->ee[ijkv] - v_stat->xx[ijkv] - v_stat->bee[ijkv] + v_stat->bxx[ijkv]);

        t_nor  += vw[v]* G0 [ijkv];   // const: should be calculated initially
        t_norb += vw[v]* G0b[ijkv];
    }

    rv[0] = t_surv, rv[1] = t_survb;
    rv[2] = t_avgP, rv[3] = t_avgPb;
    rv[4] = t_nor,  rv[5]  =  t_norb;
    rv[6] = t_aM01, rv[7]  =  t_aM02, rv[8] = t_aM03;
    rv[9] = t_maxdP;

#ifdef COSENU_MPI
    if (!myrank) MPI_Reduce(MPI_IN_PLACE, &rv[0], 10, MPI_DOUBLE, MPI_SUM, 0, CartCOMM);
    else         MPI_Reduce(&rv[0],       &rv[0], 10, MPI_DOUBLE, MPI_SUM, 0, CartCOMM);
#endif

    if (!myrank) {

        rv[0] /= rv[4];  rv[2] /= rv[4];
        rv[1] /= rv[5];  rv[3] /= rv[5];

        real aM0    = std::sqrt(rv[6]*rv[6]+rv[7]*rv[7]+rv[8]*rv[8]) * ds_L;
        real ELNe  = std::abs(n_nue0[0]*(1.0-rv[0]) - n_nue0[1]*(1.0-rv[1])) / (n_nue0[0]+n_nue0[1]); /// !!!
        //real ELNe2 = std::abs(1.0*(1.0-surv) - 0.9*(1-survb)) / (1.9);
        real Lex = rv[8] * ds_L;

        printf("T= %15f ", phy_time);
#ifdef ADV_TEST
        printf(" I1= %5.4e I2= %5.4e\n", rv[10]/rv[4], rv[11]/rv[4]);
#else
        //printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e lN= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, aM03);
        printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e ELNe= %g Lex= %g\n",rv[9],rv[0],rv[1],rv[2],rv[3],aM0, ELNe, Lex);
#endif
        anafile << phy_time << std::setprecision(13) << " " << rv[9] << " " 
            << rv[0] << " " << rv[1] << " " 
            << rv[2] << " " << rv[3] << " " 
            << aM0 << " " << Lex  << " " << ELNe << " " << endl;

        assert(rv[9] <10 && "MaxdP blowup!\n");

    }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::renormalize(const FieldVar* __restrict v0) {

    PARFORALL(i,j,k,v)   {
        int ijkv = idx(i,j,k,v);
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

