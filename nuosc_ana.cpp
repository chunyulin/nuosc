#include "nuosc_class.h"

void NuOsc::eval_extrinsics(const FieldVar* __restrict v0) {

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

        dN [ij] = (v0->ee [ij] + v0->xx [ij]);
        dN [ij] = (dN [ij] - G0[ij])/dN [ij];   // relative difference of (ee+xx)
        dNb[ij] = (v0->bee[ij] + v0->bxx[ij]);
        dNb[ij] = (dNb [ij] - G0b[ij])/dNb [ij] ;

        dP [ij] = std::abs( 1.0 - sqrt(P1 [ij]*P1 [ij]+P2 [ij]*P2 [ij]+P3 [ij]*P3 [ij]) );
        dPb[ij] = std::abs( 1.0 - sqrt(P1b[ij]*P1b[ij]+P2b[ij]*P2b[ij]+P3b[ij]*P3b[ij]) );
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
    res.std = sqrt( sum2/(nz*nv) - res.avg*res.avg  );

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
    res.std = sqrt( sum2/(nz*nv) - res.avg*res.avg  );

    return res;
}

void NuOsc::analysis() {

    eval_extrinsics(v_stat);

    real maxdP = 0.0;
    real maxdN = 0.0;
    real avgP  = 0.0, avgPb = 0.0;
    real surv  = 0.0, survb  = 0.0;
    real s0  = 0.0, s1  = 0.0;
    real nor  = 0.0, norb = 0.0;
    real aM01 = 0.0, aM02 = 0.0, aM03 = 0.0;
    //real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;

    // integral over (vz,z): assume dz=dy=const.
#pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP,maxdN) collapse(COLLAPSE_LOOP)
    FORALL(i,j,v)  {
        auto ij = idx(i,j,v);

        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        maxdP = std::max( std::max(maxdP,dP[ij]), dPb[ij]);
        //maxdN = std::max( std::max(maxdN,dN[ij]), dNb[ij]);  // What's this?

        surv  += vw[v]* v_stat->ee [ij];
        survb += vw[v]* v_stat->bee[ij];

        avgP  +=  std::abs( 1.0 - sqrt(P1 [ij]*P1 [ij]+P2 [ij]*P2 [ij]+P3 [ij]*P3 [ij]) ) * G0 [ij] * vw[v];
        avgPb +=  std::abs( 1.0 - sqrt(P1b[ij]*P1b[ij]+P2b[ij]*P2b[ij]+P3b[ij]*P3b[ij]) ) * G0b[ij] * vw[v];

        // M0
        aM01  += vw[v]* ( v_stat->ex_re[ij] - v_stat->bex_re[ij]);                                 // = P1[ij]*G0[ij] - P1b[ij]*G0b[ij];
        aM02  += vw[v]* (-v_stat->ex_im[ij] - v_stat->bex_im[ij]);                                 // = P2[ij]*G0[ij] - P2b[ij]*G0b[ij];
        aM03  += vw[v]* 0.5*(v_stat->ee[ij] - v_stat->xx[ij] - v_stat->bee[ij] + v_stat->bxx[ij]); // = P3[ij]*G0[ij] - P3b[ij]*G0b[ij], which is also the net e-x lepton number;
        // M1
        //aM11  += vw[v]* vz[i]* (v_stat->ex_re[ij] - v_stat->bex_re[ij]);
        //aM12  += vw[v]* vz[i]* (v_stat->ex_im[ij] + v_stat->bex_im[ij]);
        //aM13  += vw[v]* vz[i]* 0.5*(v_stat->ee[ij] - v_stat->xx[ij] - v_stat->bee[ij] + v_stat->bxx[ij]);

        nor  += vw[v]* G0 [ij];   // const: should be calculated initially
        norb += vw[v]* G0b[ij];
    }
    surv  /= nor;
    survb /= norb;
    avgP  /= nor;
    avgPb /= norb;
    real aM0    = sqrt(aM01*aM01+aM02*aM02+aM03*aM03) * dz;
    //real aM1    = sqrt(aM11*aM11+aM12*aM12+aM13*aM13)/nor;

    printf("T= %15f ", phy_time);
    printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e lN= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, aM03);

    anafile << phy_time << std::setprecision(13) << " " << maxdP << " " 
        << surv << " " << survb << " " 
        << avgP << " " << avgPb << " " 
        << aM0 << " " << aM03  << endl;
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

void NuOsc::snapshot(const int t) {
    char filename[32];
    sprintf(filename,"P_%05d.bin", t);

    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " << filename << endl;

#ifdef COSENU2D
    int size = (ny+2*gy)*(nz+2*gz)*nv;
#else
    int size = (nz+2*gz)*nv;
#endif

    FieldVar *v = v_stat; 

    outfile.write((char *) &phy_time,  sizeof(real));
    outfile.write((char *) &z0,  sizeof(real));
    outfile.write((char *) &z1,  sizeof(real));
    outfile.write((char *) &nz,  sizeof(int));
    outfile.write((char *) &nv,  sizeof(int));
    outfile.write((char *) &gz,  sizeof(int));
    /*
       outfile.write((char *) v->ee,     size*sizeof(real));
       outfile.write((char *) v->xx,     size*sizeof(real));
       outfile.write((char *) v->ex_re,  size*sizeof(real));
       outfile.write((char *) v->ex_im,  size*sizeof(real));
       outfile.write((char *) v->bee,    size*sizeof(real));
       outfile.write((char *) v->bxx,    size*sizeof(real));
       outfile.write((char *) v->bex_re, size*sizeof(real));
       outfile.write((char *) v->bex_im, size*sizeof(real));
       */
    outfile.write((char *) P1,  size*sizeof(real));
    outfile.write((char *) P2,  size*sizeof(real));
    outfile.write((char *) P3,  size*sizeof(real));

    outfile.close();
    printf("		Write %d x %d into %s\n", nv, nz+2*gz, filename);
}

void NuOsc::checkpoint(const int t) {
    char cptmetafn[32], cptfn[32];;
    sprintf(cptfn,    "cpt%05d.bin", t);
    sprintf(cptmetafn ,"cpt%05d.meta", t);

    std::ofstream cptmeta, cpt;
    cptmeta.open(cptmetafn, std::ofstream::out | std::ofstream::trunc);
    if(!cptmeta) cout << "*** Open fails: " << cptmetafn << endl;
    cpt.open(cptfn, std::ofstream::out | std::ofstream::trunc);
    if(!cpt) cout << "*** Open fails: " << cptfn << endl;

#ifdef COSENU2D
    int size = (ny+2*gy)*(nz+2*gz)*nv;
#else
    int size = (nz+2*gz)*nv;
#endif


    cptmeta << phy_time << " " << nz << " " << nv << " " << z0 << " " << z1 << " " << gz << endl
        << CFL << " " << dt << " " << dz << " " << mu << " " << renorm << " " << ko << endl;
    cptmeta.close();

    FieldVar *v = v_stat; 
    cpt.write((char *) v->ee,     size*sizeof(real));
    cpt.write((char *) v->xx,     size*sizeof(real));
    cpt.write((char *) v->ex_re,  size*sizeof(real));
    cpt.write((char *) v->ex_im,  size*sizeof(real));
    cpt.write((char *) v->bee,    size*sizeof(real));
    cpt.write((char *) v->bxx,    size*sizeof(real));
    cpt.write((char *) v->bex_re, size*sizeof(real));
    cpt.write((char *) v->bex_im, size*sizeof(real));
    cpt.close();
    printf("		Write %d x %d into %s\n", nv, nz+2*gz, cptfn);
}

