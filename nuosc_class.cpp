#include "nuosc_class.h"

inline void swap(FieldVar **a, FieldVar **b) { FieldVar *tmp = *a; *a = *b; *b = tmp; }
inline real random_amp(real a) { return a * rand() / RAND_MAX; }
template <typename T> int sgn(T val) {    return (T(0) < val) - (val < T(0));   }

// for init data
inline real eps_c(real z, real z0, real eps0, real lzpt){return eps0*exp(-(z-z0)*(z-z0)/lzpt);}
inline real eps_r(real z, real z0, real eps0) {return eps0*rand()/RAND_MAX;}
double g(double v, double v0, double sigma){
    double exponant = (v-v0)*(v-v0)/(2.0*sigma*sigma);
    double N = sigma*sqrt(M_PI/2.0)*(erf((1.0+v0)/sigma/sqrt(2.0))+erf((1.0-v0)/sigma/sqrt(2.0)));
    return exp(-exponant)/N;
}

//inline real eps (real z, real z0) { return 0.1*exp( -(z-z0)*(z-z0) / 50.); }
//inline real eps_(real z, real z0) { real e = eps(z, z0); return sqrt(1.0 - e*e); }
//double g(double v, double v0, double sigma, double N){
//    double exponant = (v-v0)*(v-v0)/(2.0*sigma*sigma);
//    return exp(-exponant)/N;
//}


void NuOsc::fillInitValue(real f0, real alpha, real lnue, real lnueb, int ipt, real eps0, real lzpt) {

    printf("   Init data: eps = %g  alpha = %f\n", eps0, alpha);

    //#pragma omp parallel for collapse(2)
    //#pragma acc parallel loop collapse(2), default(present)  // copyin
    for (int j=0;j<nz; j++) {
        for (int i=0;i<nvz; i++) {
            G0 [idx(i,j)] =         g(vz[i], 1.0, lnue )*0.5;
            G0b[idx(i,j)] = alpha * g(vz[i], 1.0, lnueb)*0.5;
        }
    }

    //#pragma acc update device (G0[], G0b[])


    //#pragma omp parallel for collapse(2)
    //#pragma acc parallel loop collapse(2) default(present)
    for (int j=0;j<nz; j++) {
        for (int i=0;i<nvz; i++) {
            real tmp;
            if(ipt==0) { tmp = eps_c(Z[j],0.0,eps0,lzpt); }
            if(ipt==1) { tmp = eps_r(Z[j],0.0,eps0); }
            real p3o = sqrt(1.0-tmp*tmp);
            v_stat->ee    [idx(i,j)] = G0[idx(i,j)]*(1.0+p3o); //sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
            v_stat->xx    [idx(i,j)] = G0[idx(i,j)]*(1.0-p3o);
            v_stat->ex_re [idx(i,j)] = G0[idx(i,j)]*tmp;
            v_stat->ex_im [idx(i,j)] = 0.0; //random_amp(0.001);
            v_stat->bee   [idx(i,j)] = G0b[idx(i,j)]*(1.0+p3o);
            v_stat->bxx   [idx(i,j)] = G0b[idx(i,j)]*(1.0-p3o);
            v_stat->bex_re[idx(i,j)] = G0b[idx(i,j)]*tmp;
            v_stat->bex_im[idx(i,j)] = 0.0; //random_amp(0.001);
        }
    }

    // Init boundary
#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif

}


void NuOsc::write_bin(const int t) {
    char filename[32];
    sprintf(filename,"stat_%04d.bin", t);

    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }

    FieldVar *v = v_stat; 

    outfile.write((char *) &phy_time,  sizeof(real));
    outfile.write((char *) &vz[0],     sizeof(real));
    outfile.write((char *) &vz[nvz-1], sizeof(real));
    outfile.write((char *) &Z[0],      sizeof(real));
    outfile.write((char *) &Z[nz-1],   sizeof(real));
    outfile.write((char *) &nz,  sizeof(int));
    outfile.write((char *) &nvz, sizeof(int));
    outfile.write((char *) &gz,  sizeof(int));
    outfile.write((char *) v->ee,     (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->xx,     (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->ex_re,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->ex_im,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bee,    (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bxx,    (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bex_re, (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bex_im, (nz+2*gz)*nvz*sizeof(real));
    outfile.close();
    printf("		Write %d x %d into %s\n", nvz, nz+2*gz, filename);
}


void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for collapse(2)
#pragma acc parallel loop collapse(2)
    for (int j=0;j<gz; j++)
        for (int i=0;i<nvz; i++) {
            //lower side
            in->ee    [idx(i,-j-1)] = in->ee    [idx(i,nz-j-1)];
            in->xx    [idx(i,-j-1)] = in->xx    [idx(i,nz-j-1)];
            in->ex_re [idx(i,-j-1)] = in->ex_re [idx(i,nz-j-1)];
            in->ex_im [idx(i,-j-1)] = in->ex_im [idx(i,nz-j-1)];
            in->bee   [idx(i,-j-1)] = in->bee   [idx(i,nz-j-1)];
            in->bxx   [idx(i,-j-1)] = in->bxx   [idx(i,nz-j-1)];
            in->bex_re[idx(i,-j-1)] = in->bex_re[idx(i,nz-j-1)];
            in->bex_im[idx(i,-j-1)] = in->bex_im[idx(i,nz-j-1)];
            //upper side
            in->ee    [idx(i,nz+j)] = in->ee    [idx(i,j)];
            in->xx    [idx(i,nz+j)] = in->xx    [idx(i,j)];
            in->ex_re [idx(i,nz+j)] = in->ex_re [idx(i,j)];
            in->ex_im [idx(i,nz+j)] = in->ex_im [idx(i,j)];
            in->bee   [idx(i,nz+j)] = in->bee   [idx(i,j)];
            in->bxx   [idx(i,nz+j)] = in->bxx   [idx(i,j)];
            in->bex_re[idx(i,nz+j)] = in->bex_re[idx(i,j)];
            in->bex_im[idx(i,nz+j)] = in->bex_im[idx(i,j)];
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { }

void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    //#pragma omp target teams distribute parallel for collapse(2)
    //1.#pragma acc parallel loop collapse(2)
    //2.#pragma acc parallel loop gang
    //3.#pragma acc parallel loop gang
#pragma omp parallel for collapse(2)
#pragma acc parallel loop gang
    for (int j=0;j<nz; j++)
        //2.#pragma acc parallel loop vector
        //3.#pragma acc loop worker
#pragma acc loop vector
        for (int i=0;i<nvz; i++) {
            real *ee    = &(in->ee    [idx(i,j)]);
            real *xx    = &(in->xx    [idx(i,j)]);
            real *exr   = &(in->ex_re [idx(i,j)]);
            real *exi   = &(in->ex_im [idx(i,j)]);
            real *bee   = &(in->bee   [idx(i,j)]);
            real *bxx   = &(in->bxx   [idx(i,j)]);
            real *bexr  = &(in->bex_re[idx(i,j)]);
            real *bexi  = &(in->bex_im[idx(i,j)]);

            // 1) prepare terms for -i [H0, rho]
            out->ee    [idx(i,j)] = -pmo *   2*st*exi [0];
            out->xx    [idx(i,j)] =  pmo *   2*st*exi [0];
            out->ex_re [idx(i,j)] = -pmo *   2*ct*exi [0];
            out->ex_im [idx(i,j)] =  pmo * ( 2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [idx(i,j)] = -pmo *   2*st*bexi[0];
            out->bxx   [idx(i,j)] =  pmo *   2*st*bexi[0];
            out->bex_re[idx(i,j)] = -pmo *   2*ct*bexi[0];
            out->bex_im[idx(i,j)] =  pmo * ( 2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );

#ifndef ADVEC_OFF
#if defined(ADVEC_CENTER_FD)
            // 2) advection term: (central FD)
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor = -vz[i]/(12*dz);
//#define ADV_FD(x)  (  (x[-2]-x[2]) - 8.0*(x[-1]-x[1]) )
#define ADV_FD(x)  (  (x[-2*nvz]-x[2*nvz]) - 8.0*(x[-1*nvz]-x[1*nvz]) )
#elif defined(ADVEC_UPWIND)
            int sv = sgn(vz[i]);
            real factor = -sv*vz[i]/(12*dz);
#define ADV_FD(x)  ( -4*x[-3*sv*nvz] + 18*x[-2*sv*nvz] - 36*x[-sv*nvz] + 22*x[0] )
#else
            // advection term: (4-th order lopsided finite differencing)
            int sv = sgn(vz[i]);
            real factor = sv*vz[i]/(12*dz);
#define ADV_FD(x)  ( x[-3*sv*nvz] - 6*x[-2*sv*nvz] + 18*x[-sv*nvz] - 10*x[0] - 3*x[sv*nvz] )
#endif
            out->ee    [idx(i,j)] += factor * ADV_FD(ee);
            out->xx    [idx(i,j)] += factor * ADV_FD(xx);
            out->ex_re [idx(i,j)] += factor * ADV_FD(exr);
            out->ex_im [idx(i,j)] += factor * ADV_FD(exi);
            out->bee   [idx(i,j)] += factor * ADV_FD(bee);
            out->bxx   [idx(i,j)] += factor * ADV_FD(bxx);
            out->bex_re[idx(i,j)] += factor * ADV_FD(bexr);
            out->bex_im[idx(i,j)] += factor * ADV_FD(bexi);
#undef ADV_FD
#endif

#ifdef NVTX
    nvtxRangePush("mu-loop");
#endif
            if (mu>0.0) {
                // 3) interaction terms: vz-integral with a simple trapezoidal rule (can be optimized later)
                real Iee    = 0;
                real Ixx    = 0;
                real Iexr   = 0;
                real Iexi   = 0;
                real Ibee   = 0;
                real Ibxx   = 0;
                real Ibexr  = 0;
                real Ibexi  = 0;

                //3.#pragma acc loop vector  // implicit reduction detected
                //#pragma acc loop vector
#pragma acc loop
                for (int k=0;k<nvz; k++) {   // vz' integral
                    real eep    = (in->ee    [idx(k,j)]);
                    real xxp    = (in->xx    [idx(k,j)]);
                    real expr   = (in->ex_re [idx(k,j)]);
                    real expi   = (in->ex_im [idx(k,j)]);
                    real beep   = (in->bee   [idx(k,j)]);
                    real bxxp   = (in->bxx   [idx(k,j)]);
                    real bexpr  = (in->bex_re[idx(k,j)]);
                    real bexpi  = (in->bex_im[idx(k,j)]);

                    // terms for -i* mu * [rho'-rho_bar', rho]
                    Iee   +=  2*vw[k]*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                    Ixx   += -2*vw[k]*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );  // = -Iee
                    Iexr  +=    vw[k]*mu* (1-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                    Iexi  +=    vw[k]*mu* (1-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                    Ibee  +=  2*vw[k]*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                    Ibxx  += -2*vw[k]*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) ); // = -Ibee
                    Ibexr +=    vw[k]*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                    Ibexi +=    vw[k]*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
                }
                // 3.1) calculate integral with simple trapezoidal rule
                out->ee    [idx(i,j)] += dv*Iee;
                out->xx    [idx(i,j)] += dv*Ixx;
                out->ex_re [idx(i,j)] += dv*Iexr;
                out->ex_im [idx(i,j)] += dv*Iexi;
                out->bee   [idx(i,j)] += dv*Ibee;
                out->bxx   [idx(i,j)] += dv*Ibxx;
                out->bex_re[idx(i,j)] += dv*Ibexr;
                out->bex_im[idx(i,j)] += dv*Ibexi;
            } // end of mu-part
#ifdef NVTX
    nvtxRangePop();
#endif

#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps = -ko/dz/64.0;
#define KO_FD(x)  ( x[-3] + x[3] - 6*(x[-2]+x[2]) + 15*(x[-1]+x[1]) - 20*x[0] )
#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps = -ko/dz/16.0;
#define KO_FD(x)  ( x[-2] + x[2] - 4*(x[-1]+x[1]) + 6*x[0] )
#endif
            out->ee    [idx(i,j)] += ko_eps * KO_FD(ee);
            out->xx    [idx(i,j)] += ko_eps * KO_FD(xx);
            out->ex_re [idx(i,j)] += ko_eps * KO_FD(exr);
            out->ex_im [idx(i,j)] += ko_eps * KO_FD(exi);
            out->bee   [idx(i,j)] += ko_eps * KO_FD(bee);
            out->bxx   [idx(i,j)] += ko_eps * KO_FD(bxx);
            out->bex_re[idx(i,j)] += ko_eps * KO_FD(bexr);
            out->bex_im[idx(i,j)] += ko_eps * KO_FD(bexi);
#undef KO_FD
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

#pragma omp parallel for collapse(2) 
#pragma acc parallel loop collapse(2)
    for (int j=0;j<nz; j++)
        for (int i=0;i<nvz; i++) {
            int k = idx(i,j);
            v0->ee    [k] = v1->ee    [k] + a * v2->ee    [k];
            v0->xx    [k] = v1->xx    [k] + a * v2->xx    [k];
            v0->ex_re [k] = v1->ex_re [k] + a * v2->ex_re [k];
            v0->ex_im [k] = v1->ex_im [k] + a * v2->ex_im [k];
            v0->bee   [k] = v1->bee   [k] + a * v2->bee   [k];
            v0->bxx   [k] = v1->bxx   [k] + a * v2->bxx   [k];
            v0->bex_re[k] = v1->bex_re[k] + a * v2->bex_re[k];
            v0->bex_im[k] = v1->bex_im[k] + a * v2->bex_im[k];
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2, const FieldVar * __restrict v3) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

#pragma omp parallel for collapse(2)
#pragma acc parallel loop collapse(2)
    for (int j=0;j<nz; j++)
        for (int i=0;i<nvz; i++) {
            int k = idx(i,j);
            v0->ee    [k] = v1->ee    [k] + a * (v2->ee    [k] + v3->ee    [k]);
            v0->xx    [k] = v1->xx    [k] + a * (v2->xx    [k] + v3->xx    [k]);
            v0->ex_re [k] = v1->ex_re [k] + a * (v2->ex_re [k] + v3->ex_re [k]);
            v0->ex_im [k] = v1->ex_im [k] + a * (v2->ex_im [k] + v3->ex_im [k]);
            v0->bee   [k] = v1->bee   [k] + a * (v2->bee   [k] + v3->bee   [k]);
            v0->bxx   [k] = v1->bxx   [k] + a * (v2->bxx   [k] + v3->bxx   [k]);
            v0->bex_re[k] = v1->bex_re[k] + a * (v2->bex_re[k] + v3->bex_re[k]);
            v0->bex_im[k] = v1->bex_im[k] + a * (v2->bex_im[k] + v3->bex_im[k]);
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::step_rk4() {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    //Step-1
#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif
    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5*dt, v_rhs);

    //Step-2
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, 0.5*dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-3
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
#endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-4
#ifdef BC_PERI
    updatePeriodicBoundary(v_pre);
#else
    updateInjetOpenBoundary(v_pre);
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

