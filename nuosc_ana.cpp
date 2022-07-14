#include "nuosc_class.h"

void NuOsc::eval_extrinsics(const FieldVar* __restrict v0) {

    PARFORALL(i,p,v)   {
        int ipv=idx(i,p,v);
        real iG  = 1.0 / G0[ipv];
        real iGb = 1.0 / G0b[ipv];
        P1 [ipv] =   2.0*v0->ex_re[ipv] * iG;
        P2 [ipv] = - 2.0*v0->ex_im[ipv] * iG;
        P3 [ipv] = (v0->ee[ipv] - v0->xx[ipv])*iG;
        P1b[ipv] =  2.0*v0->bex_re[ipv] * iGb;
        P2b[ipv] =  2.0*v0->bex_im[ipv] * iGb;
        P3b[ipv] = (v0->bee[ipv] - v0->bxx[ipv]) * iGb;

        dN [ipv] = (v0->ee [ipv] + v0->xx [ipv]);
        dN [ipv] = (dN [ipv] - G0[ipv])/dN [ipv];   // relative difference of (ee+xx)
        dNb[ipv] = (v0->bee[ipv] + v0->bxx[ipv]);
        dNb[ipv] = (dNb [ipv] - G0b[ipv])/dNb [ipv] ;

        dP [ipv] = std::abs( 1.0 - std::sqrt(P1 [ipv]*P1 [ipv]+P2 [ipv]*P2 [ipv]+P3 [ipv]*P3 [ipv]) );
        dPb[ipv] = std::abs( 1.0 - std::sqrt(P1b[ipv]*P1b[ipv]+P2b[ipv]*P2b[ipv]+P3b[ipv]*P3b[ipv]) );
    }
}

void NuOsc::renormalize(FieldVar* const __restrict v0) {

    //cout << "Not functioned yet!" << endl;
    //assert(0);
    PARFORALL(i,p,v)   {
        int ipv=idx(i,p,v);
        real iG = 1.0 / G0[ipv];
        real iGb = 1.0 / G0b[ipv];
        real P1  =   2.0* v0->ex_re[ipv] * iG;
        real P2  = - 2.0* v0->ex_im[ipv] * iG;
        real P3  = (v0->ee[ipv] - v0->xx[ipv])*iG;
        real P1b =   2.0* v0->bex_re[ipv] * iGb;
        real P2b =   2.0* v0->bex_im[ipv] * iGb;
        real P3b = (v0->bee[ipv] - v0->bxx[ipv]) * iGb;
        real iP   = 1.0/std::sqrt(P1*P1+P2*P2+P3*P3);
        real iPb  = 1.0/std::sqrt(P1b*P1b+P2b*P2b+P3b*P3b);
        real tmp  = 0.5*iP *(P3) *G0 [ipv];
        real tmpb = 0.5*iPb*(P3b)*G0b[ipv];

        v0->ee    [ipv]  = G0[ipv]+tmp;
        v0->xx    [ipv]  = G0[ipv]-tmp;
        v0->ex_re [ipv] *= iP;
        v0->ex_im [ipv] *= iP;
        v0->bee   [ipv]  = G0b[ipv]+tmpb;
        v0->bxx   [ipv]  = G0b[ipv]-tmpb;
        v0->bex_re[ipv] *= iPb;
        v0->bex_im[ipv] *= iPb;
    }
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
#ifdef ADV_TEST
    real I1  = 0.0, I2 = 0.0;
#endif
    //real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;

    // integral over (vz,z): assume dz=dy=const.
#pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP,maxdN) collapse(COLLAPSE_LOOP)
#pragma acc parallel loop reduction(+:avgP,avgPb,aM01,aM02,aM03,nor,norb,surv,survb) reduction(max:maxdP,maxdN) collapse(COLLAPSE_LOOP)
    FORALL(i,p,v)  {
        auto ipv = idx(i,p,v);

        //if (dP>maxdP || dPb>maxdP) {maxi=i;maxj=j;}
        maxdP = std::max( std::max(maxdP,dP[ipv]), dPb[ipv]);
        //maxdN = std::max( std::max(maxdN,dN[ipv]), dNb[ipv]);  // What's this?

#ifdef ADV_TEST
        I1 += vw[v]*w[p]* v_stat->ee [ipv]* v_stat->ee [ipv];
        I2 += vw[v]*w[p]* v_stat->ee [ipv];
#endif
        surv  += vw[v]*w[p] * v_stat->ee [ipv];
        survb += vw[v]*w[p] * v_stat->bee[ipv];

        avgP  += vw[v]*w[p] * std::abs( 1.0 - std::sqrt(P1 [ipv]*P1 [ipv]+P2 [ipv]*P2 [ipv]+P3 [ipv]*P3 [ipv]) ) * G0 [ipv];
        avgPb += vw[v]*w[p] * std::abs( 1.0 - std::sqrt(P1b[ipv]*P1b[ipv]+P2b[ipv]*P2b[ipv]+P3b[ipv]*P3b[ipv]) ) * G0b[ipv];

        // M0 * w[p]
        aM01  += vw[v]*w[p] * ( v_stat->ex_re[ipv] - v_stat->bex_re[ipv]);                                 // = P1[ipv]*G0[ipv] - P1b[ipv]*G0b[ipv];
        aM02  += vw[v]*w[p] * (-v_stat->ex_im[ipv] - v_stat->bex_im[ipv]);                                 // = P2[ipv]*G0[ipv] - P2b[ipv]*G0b[ipv];
        aM03  += vw[v]*w[p] * 0.5*(v_stat->ee[ipv] - v_stat->xx[ipv] - v_stat->bee[ipv] + v_stat->bxx[ipv]); // = P3[ipv]*G0[ipv] - P3b[ipv]*G0b[ipv], which is also the net e-x lepton number;
        // M1
        //aM11  += vw[v]* vz[i]* (v_stat->ex_re[ipv] - v_stat->bex_re[ipv]);
        //aM12  += vw[v]* vz[i]* (v_stat->ex_im[ipv] + v_stat->bex_im[ipv]);
        //aM13  += vw[v]* vz[i]* 0.5*(v_stat->ee[ipv] - v_stat->xx[ipv] - v_stat->bee[ipv] + v_stat->bxx[ipv]);

        nor  += vw[v]*w[p] * G0 [ipv];   // const: should be calculated initially
        norb += vw[v]*w[p] * G0b[ipv];
    }
    surv  /= nor;
    survb /= norb;
    avgP  /= nor;
    avgPb /= norb;
    real aM0    = std::sqrt(aM01*aM01+aM02*aM02+aM03*aM03) * dz*0.5 / (z1-z0);
    //real aM1    = std::sqrt(aM11*aM11+aM12*aM12+aM13*aM13)/nor;
    real ELNe  = std::abs(n_nue0*(1.0-surv) - n_nueb0*(1.0-survb)) / (n_nue0 + n_nueb0);
    real ELNe2 = std::abs(1.0*(1.0-surv) - 0.9*(1-survb)) / (1.9);

    printf("T= %15f ", phy_time);
    //printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e lN= %g\n",maxdP,surv,survb,avgP,avgPb,aM0, aM03);
#ifdef ADV_TEST
    printf(" I1= %5.4e  I2= %5.4e\n", I1/nor, I2/nor);
#else
    printf(" |dP|max= %5.4e surb= %5.4e %5.4e conP= %5.4e %5.4e |M0|= %5.4e ELNe= %g %g\n",maxdP,surv,survb,avgP,avgPb,aM0, ELNe, ELNe2);
#endif

    anafile << phy_time << std::setprecision(13) << " " << maxdP << " "
            << surv << " " << survb << " "
            << avgP << " " << avgPb << " "
            << aM0 << " " << aM03  << " " << ELNe << " " << ELNe2 << endl;
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
            //   outfile << vz[i] << " " << Z[j] << " " << v_stat->ee[ij] << " " << v_stat->ex_re[ij] << " " << v_stat->ex_im[ij] << endl;
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

    int size = (nz)*nv;

    FieldVar *v = v_stat; 

    outfile.write((char *) &phy_time,  sizeof(real));
    outfile.write((char *) &z0,  sizeof(real));
    outfile.write((char *) &z1,  sizeof(real));
    outfile.write((char *) &nz,  sizeof(int));
    outfile.write((char *) &nv,  sizeof(int));
    //outfile.write((char *) &gz,  sizeof(int));
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
    printf("		Write %d x %d into %s\n", nv, nz, filename);
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

    int size = nz*nv;


    cptmeta << phy_time << " " << nz << " " << nv << " " << z0 << " " << z1 << endl
        << CFL << " " << dt << " " << dz << " " << mu << " " << renorm << endl;
    cptmeta.close();

    FieldVar *v = v_stat; 
    cpt.write((char *) v->ee.data(),     size*sizeof(real));
    cpt.write((char *) v->xx.data(),     size*sizeof(real));
    cpt.write((char *) v->ex_re.data(),  size*sizeof(real));
    cpt.write((char *) v->ex_im.data(),  size*sizeof(real));
    cpt.write((char *) v->bee.data(),    size*sizeof(real));
    cpt.write((char *) v->bxx.data(),    size*sizeof(real));
    cpt.write((char *) v->bex_re.data(), size*sizeof(real));
    cpt.write((char *) v->bex_im.data(), size*sizeof(real));
    cpt.close();
    printf("		Write %d x %d into %s\n", nv, nz, cptfn);
}

FieldStat NuOsc::_analysis_v(const real var[]) {

    FieldStat res;

    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;
#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax) collapse(COLLAPSE_LOOP)
    FORALL(i,p,v)  {
        real val = var[idx(i,p,v)];
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
    FORALL(i,p,v)  {
        auto ij = idx(i,p,v);
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

