#include "nuosc_class.h"

void NuOsc::eval_conserved(const FieldVar* __restrict v0) {

    PARFORALL(i,j,v)   {
        int ij=idx(i,j,v);
        real iG  = 1.0 / G0[ij];
        real iGb = 1.0 / G0b[ij];
        P1 [ij] =   2.0*v0->ex_re[ij] * iG;
        P2 [ij] = - 2.0*v0->ex_im[ij] * iG;
        P3 [ij] = (v0->ee[ij] - v0->xx[ij])*iG;
        P1b[ij] =  2.0*v0->bex_re[ij] * iGb;
        P2b[ij] =  2.0*v0->bex_im[ij] * iGb;
        P3b[ij] = (v0->bee[ij] - v0->bxx[ij]) * iGb;

        relN [ij] = (v0->ee [ij] + v0->xx [ij]);
        relN [ij] = (relN [ij] - G0[ij])/relN [ij];   // relative difference of (ee+xx)
        relNb[ij] = (v0->bee[ij] + v0->bxx[ij]);
        relNb[ij] = (relNb [ij] - G0b[ij])/relNb [ij] ;

        relP [ij] = sqrt(P1 [ij]*P1 [ij]+P2 [ij]*P2 [ij]+P3 [ij]*P3 [ij]);
        relP [ij] = (1.0 - relP [ij])/relP [ij];       // -relP actually to save one fabs(). Note init |P|=1
        relPb[ij] = sqrt(P1b[ij]*P1b[ij]+P2b[ij]*P2b[ij]+P3b[ij]*P3b[ij]);
        relPb[ij] = (1.0 - relPb[ij])/relPb[ij];
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
        real iP   = 1.0/sqrt(P1*P1+P2*P2+P3*P3);
        real iPb  = 1.0/sqrt(P1b*P1b+P2b*P2b+P3b*P3b);
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

FieldStat NuOsc::_analysis_v(const real var[]) {

    FieldStat res;

    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;

#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax) collapse(3)
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
    res.std = sqrt( sum2/(nz*nv) - res.avg*res.avg  );

    return res;
}

FieldStat NuOsc::_analysis_c(const real vr[], const real vi[]) {

    FieldStat res;
    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;

#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax) collapse(3)
    FORALL(i,j,v)  {
        int ij = idx(i,j,v);
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
    res.std = sqrt( sum2/(nz*nv) - res.avg*res.avg  );

    return res;
}


void NuOsc::analysis() {

    eval_conserved(v_stat);

    real maxrelP = 0.0;
    real maxrelN = 0.0;
    real avgP  = 0.0;
    real avgPb = 0.0;
    real surv  = 0.0;
    real survb = 0.0;
    real norP  = 0.0;
    real norPb = 0.0;
    real aM01 = 0.0, aM02 = 0.0, aM03 = 0.0;
    //real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;

    // integral over (vz,z): assume dz=dy=const.
#pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,norP,norPb) reduction(max:maxrelP,maxrelN) collapse(3)
    FORALL(i,j,v)  {
        int ij = idx(i,j,v);

        //if (relP>maxrelP || relPb>maxrelP) {maxi=i;maxj=j;}
        maxrelP = std::max( std::max(maxrelP,relP[ij]), relPb[ij]);
        //maxrelN = std::max( std::max(maxrelN,relN[ij]), relNb[ij]);  // What's this?

        surv  += vw[v]* v_stat->ee [ij];
        survb += vw[v]* v_stat->bee[ij];

        avgP  +=  ( 1.0 - sqrt(P1 [ij]*P1 [ij]+P2 [ij]*P2 [ij]+P3 [ij]*P3 [ij]) ) * G0 [ij] * vw[v];
        avgPb +=  ( 1.0 - sqrt(P1b[ij]*P1b[ij]+P2b[ij]*P2b[ij]+P3b[ij]*P3b[ij]) ) * G0b[ij] * vw[v];

        // M0
        aM01  += vw[v]*  (v_stat->ex_re[ij] - v_stat->bex_re[ij]);                                  // P1[ij]*G0[ij] - P1b[ij]*G0b[ij];
        aM02  += vw[v]* (-v_stat->ex_im[ij] - v_stat->bex_im[ij]);                                  // P2[ij]*G0[ij] - P2b[ij]*G0b[ij];
        aM03  += vw[v]* 0.5*(v_stat->ee[ij] - v_stat->xx[ij] - v_stat->bee[ij] + v_stat->bxx[ij]);   // P3[ij]*G0[ij] - P3b[ij]*G0b[ij];
        // M1
        //aM11  += vw[v]* vz[i]* (v_stat->ex_re[ij] - v_stat->bex_re[ij]);
        //aM12  += vw[v]* vz[i]* (v_stat->ex_im[ij] + v_stat->bex_im[ij]);
        //aM13  += vw[v]* vz[i]* 0.5*(v_stat->ee[ij] - v_stat->xx[ij] - v_stat->bee[ij] + v_stat->bxx[ij]);
        
        norP  += vw[v]* G0 [ij];   // const: should be calculated initially
        norPb += vw[v]* G0b[ij];
    }
    surv  /= norP;
    survb /= norPb;
    avgP  /= norP;
    avgPb /= norPb;
    real aM0    = sqrt(aM01*aM01+aM02*aM02+aM03*aM03);
    //real aM1    = sqrt(aM11*aM11+aM12*aM12+aM13*aM13)/nor;
    
    printf("T= %15f ", phy_time);
    printf(" |dP|max= %5.4e   surb= %5.4e %5.4e  conP= %5.4e %5.4e  |M0|= %5.4e\n",maxrelP,surv,survb,avgP,avgPb,aM0);

    anafile << phy_time << std::setprecision(13) << " " << maxrelP << " " 
                        << surv << " " << survb << " " 
                        << avgP << " " << avgPb << " " 
                        << aM0  << endl;
}




void NuOsc::output_detail(const char* filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }
    outfile << "#phy_time=" << phy_time << endl;
    int iskip=nz/100;
    for(int i=0;i<nv;i++){
        for(int j=0;j<nz;j=j+iskip){
            int ij=idx(i,j,i);   // FIXME
            outfile << vz[i] << " " << Z[j] << " " << P1 << " " << P2 << " " << P3 << endl;
            //        outfile << vz[i] << " " << Z[j] << " " << v_stat->ee[ij] << " " << v_stat->ex_re[ij] << " " << v_stat->ex_im[ij] << endl;
        }
        outfile << endl;
    }
}


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
