#include "nuosc_class.h"

void NuOsc::eval_conserved(const FieldVar* __restrict v0) {
#ifdef NVTX
    nvtxRangePush("eval_conserved");
#endif

    PARFORALL(i,j,v)   {
        uint ijv = idx(i,j,v);
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
        int ijv = idx(i,j,v);

#ifdef ADV_TEST
        I1 += vw[v]* v_stat->ee [ijv];
        I2 += vw[v]* v_stat->ee [ijv]* v_stat->ee [ijv];
#endif
                
        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        maxdP = std::max( maxdP, std::max(dP[ijv], dPb[ijv]));
        //maxdN = std::max( std::max(maxdN,dN[ijv]), dNb[ijv]);  // What's this?

        surv  += vw[v]* v_stat->ee [ijv];
        survb += vw[v]* v_stat->bee[ijv];

        avgP  += vw[v] * G0 [ijv] * std::abs( 1.0 - std::sqrt(P1 [ijv]*P1 [ijv]+P2 [ijv]*P2 [ijv]+P3 [ijv]*P3 [ijv]) );
        avgPb += vw[v] * G0b[ijv] * std::abs( 1.0 - std::sqrt(P1b[ijv]*P1b[ijv]+P2b[ijv]*P2b[ijv]+P3b[ijv]*P3b[ijv]) );

        // M0
        //aM01  += vw[v]* ( v_stat->ex_re[ijv] - v_stat->bex_re[ijv]);                                 // = P1[ijv]*G0[ijv] - P1b[ijv]*G0b[ijv];
        //aM02  += vw[v]* (-v_stat->ex_im[ijv] - v_stat->bex_im[ijv]);                                 // = P2[ijv]*G0[ijv] - P2b[ijv]*G0b[ijv];
        aM03  += vw[v]* 0.5*(v_stat->ee[ijv] - v_stat->xx[ijv] - v_stat->bee[ijv] + v_stat->bxx[ijv]); // = P3[ijv]*G0[ijv] - P3b[ijv]*G0b[ijv], which is also the net e-x lepton number;
        
        // M1
        //aM11  += vw[v]* vz[i]* (v_stat->ex_re[ijv] - v_stat->bex_re[ijv]);
        //aM12  += vw[v]* vz[i]* (v_stat->ex_im[ijv] + v_stat->bex_im[ijv]);
        //aM13  += vw[v]* vz[i]* 0.5*(v_stat->ee[ijv] - v_stat->xx[ijv] - v_stat->bee[ijv] + v_stat->bxx[ijv]);
        
        nor  += vw[v]* G0 [ijv];   // const: should be calculated initially
        norb += vw[v]* G0b[ijv];
    }
    surv  /= nor;
    survb /= norb;
    avgP  /= nor;
    avgPb /= norb;
    real aM0    = std::sqrt(aM01*aM01+aM02*aM02+aM03*aM03) * dx * dz / ((z1-z0) * (x1-x0));
    //real aM1    = std::sqrt(aM11*aM11+aM12*aM12+aM13*aM13)/nor;
    real ELNe  = std::abs(n_nue0*(1.0-surv) - n_nueb0*(1.0-survb)) / (n_nue0 + n_nueb0);
    //real ELNe2 = std::abs(1.0*(1.0-surv) - 0.9*(1-survb)) / (1.9);
    real Lex = aM03 * dx * dz / ((z1-z0) * (x1-x0));


    printf("T= %15f ", phy_time);
    //printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e lN= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, aM03);
#ifdef ADV_TEST
    printf(" I1= %5.4e I2= %5.4e\n", I1/nor, I2/nor);
#else
    printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e ELNe= %g Lex= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, ELNe, Lex);
#endif
    anafile << phy_time << std::setprecision(13) << " " << maxdP << " " 
                        << surv << " " << survb << " " 
                        << avgP << " " << avgPb << " " 
                        << aM0 << " " << Lex  << " " << ELNe << " " << endl;

    assert(maxdP <10 );
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::output_detail(const char* filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }
    outfile << "#phy_time=" << phy_time << endl;
    int iskip=nz/100;
    for(int i=0;i<nv;++i){
        for(int j=0;j<nz;j=j+iskip){
            int ij=idx(i,j,i);   // FIXME
            outfile << vz[i] << " " << Z[j] << " " << P1 << " " << P2 << " " << P3 << endl;
            //        outfile << vz[i] << " " << Z[j] << " " << v_stat->ee[ij] << " " << v_stat->ex_re[ij] << " " << v_stat->ex_im[ij] << endl;
        }
        outfile << endl;
    }
}

void NuOsc::renormalize(const FieldVar* __restrict v0) {

    PARFORALL(i,j,v)   {
        int ij=idx(i,j,v);
        real iG = 1.0 / G0[ij];
        real iGb = 1.0 / G0b[ij];
        real P1  =   v0->ex_re[ij] * iG;
        real P2  = - v0->ex_im[ij] * iG;
        real P3  = (v0->ee[ij] - v0->xx[ij])*iG;
        real P1b = v0->bex_re[ij] * iGb;
        real P2b = v0->bex_im[ij] * iGb;
        real P3b = (v0->bee[ij] - v0->bxx[ij]) * iGb;
        real iP   = 1.0/std::sqrt(P1*P1+P2*P2+P3*P3);
        real iPb  = 1.0/std::sqrt(P1b*P1b+P2b*P2b+P3b*P3b);
        real tmp  = iP *(P3) *G0 [ij];
        real tmpb = iPb*(P3b)*G0b[ij];

        v0->ee    [ij]  = G0[ij]+tmp;
        v0->xx    [ij]  = G0[ij]-tmp;
        v0->ex_re [ij] *= iP;
        v0->ex_im [ij] *= iP;
        v0->bee   [ij]  = G0b[ij]+tmpb;
        v0->bxx   [ij]  = G0b[ij]-tmpb;
        v0->bex_re[ij] *= iPb;
        v0->bex_im[ij] *= iPb;
    }
}


/*


void NuOsc::write_fz() {   // FIXME
#define WRITE_Z_AT(HANDLE, VAR, V_IDX) \
    HANDLE << phy_time << " "; \
    for (int i=0;i<nz; i++) HANDLE << std::setprecision(8) << VAR[idx(V_IDX, i,i)] << " "; \
    HANDLE << endl;
    // f(z) at the highest v-mode
    //WRITE_Z_AT(ee_vh,  v->ee,    nv-1)

    {
        // Pn at v = -0.5
        int at_vz = int(0.25*(nv-1));
        WRITE_Z_AT(p1_vm,   P1,  at_vz);
        WRITE_Z_AT(p2_vm,   P2,  at_vz);
        WRITE_Z_AT(p3_vm,   P3,  at_vz);
    }
    {    // Pn at v = 1
        int at_vz = nv-1;
        WRITE_Z_AT(p1_v,   P1,  at_vz);
        WRITE_Z_AT(p2_v,   P2,  at_vz);
        WRITE_Z_AT(p3_v,   P3,  at_vz);
    }
#undef WRITE_Z_AT

}

FieldStat NuOsc::_analysis_v(const real var[]) {

    FieldStat res;

    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;
#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax) collapse(COLLAPSE_LOOP)
    FORALL(i,j,v)  {
        real val = var[idx(i,j,v)];
        if (val>vmax) vmax = val;
        if (val<vmin) vmin = val;
        sum  += val;
        sum2 += val*val;
    }

    // min, max, avg, std
    res.min = vmin;
    res.max = vmax;
    res.sum = sum;
    res.avg = sum/(nz*nv);
    res.std = std::sqrt( sum2/(nz*nv) - res.avg*res.avg  );

    return res;
}

FieldStat NuOsc::_analysis_c(const real vr[], const real vi[]) {

    FieldStat res;
    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;

#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax) collapse(COLLAPSE_LOOP)
    FORALL(i,j,v)  {
        auto ij = idx(i,j,v);
        real val = vr[ij]*vr[ij]+vi[ij]*vi[ij];
        if (val>vmax) vmax = val;
        if (val<vmin) vmin = val;
        sum  += val;
        sum2 += val*val;
    }

    // min, max, avg, std
    res.min = vmin;
    res.max = vmax;
    res.sum = sum;
    res.avg = sum/(nz*nv);
    res.std = std::sqrt( sum2/(nz*nv) - res.avg*res.avg  );

    return res;
}


*/

