#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>

#define BC_PERI
//#define ADVEC_OFF
#define KO_ORD_3

using std::cout;
using std::endl;
using std::cin;

typedef double real;

typedef struct Vars {
    real* ee;
    real* ex_re;
    real* ex_im;
    real* bee;
    real* bex_re;
    real* bex_im;

    Vars(int size) {
        ee     = new real[size];
        ex_re  = new real[size];
        ex_im  = new real[size];
        bee    = new real[size];
        bex_re = new real[size];
        bex_im = new real[size];
    }
    ~Vars() {
        delete[] ee;
        delete[] ex_re;
        delete[] ex_im;
        delete[] bee;
        delete[] bex_re;
        delete[] bex_im;
    }
} FieldVar;

inline void swap(FieldVar **a, FieldVar **b) { FieldVar *tmp = *a; *a = *b; *b = tmp; }
inline real random_amp(real a) { return a * rand() / RAND_MAX; }
template <typename T> int sgn(T val) {    return (T(0) < val) - (val < T(0));   }

// for init data
inline real eps (real z, real z0) { return 0.1*exp( -(z-z0)*(z-z0) / 50.); }
inline real eps_(real z, real z0) { real e = eps(z, z0); return sqrt(1.0 - e*e); }
double g(double v, double v0, double sigma, double N){
    double exponant = (v-v0)*(v-v0)/(2.0*sigma*sigma);
    return exp(-exponant)/N;
}

class NuOsc {
    public:
        real phy_time, dt;

        // all coordinates
        real  *vz, *Z;
        // all field variables
        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;
        real* con1;

        int nvz; // Dim of vz (Vertex-center grid used.)
        int nz;  // Dim of z  (the last dimension, the one with derivatives. Cell-center grid used.)
        int gz;  // Width of z-buffer zone. 4 for 2nd-order of d/dz.
        real vz0, vz1;
        real  z0,  z1;
        real  dv,  dz;
        real CFL;
        real ko;

        const real theta = 1e-6;
        const real ct = cos(2*theta);
        const real st = sin(2*theta);
        const int  pmo = 0; // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
        real mu = 0.0;        // can be set by set_mu()

        std::ofstream anafile;

	// text output of f(z) at certain v-mode
        std::ofstream ee_vl,  ee_vh, ee_vm;
        std::ofstream exr_vl, exr_vh;
        std::ofstream exi_vl, exi_vh;
        std::ofstream con_vl, con_vh, con_vm;

        inline unsigned int idx(const int i, const int j) { return i*(nz+2*gz) + (j+gz); }    // i:vz   j:z (last index)
        inline unsigned int idz(const int j) { return j; }

        NuOsc(const int  nvz_, const int   nz_,
              const real vz0_, const real vz1_,
              const real  z0_, const real  z1_,
              const real CFL_, const real  ko_, const int gz_ = 3) : phy_time(0.), ko(ko_)  {

            vz0 = vz0_;  vz1 = vz1_;
            z0  = z0_;    z1 = z1_;
            CFL = CFL_;

            nvz = nvz_;
            nz  = nz_;
            gz  = gz_;
            int size = (nz+2*gz)*(nvz);
            vz     = new real[nvz];
            Z      = new real[nz];
            con1   = new real[size];

            v_stat = new FieldVar(size);
            v_rhs  = new FieldVar(size);
            v_pre  = new FieldVar(size);
            v_cor  = new FieldVar(size);

            dv = (vz1-vz0)/(nvz-1);      // for v as vertex-center
            //dv = (vz1-vz0)/nvz;      // for v as cell-center
            dz = (z1-z0)/nz;       // cell-center
            dt = dz*CFL;

            //for (int i=0;i<nvz; i++)	vz[i] = vz0 + (i+0.5)*dv;   // cell-center
            for (int i=0;i<nvz; i++)	vz[i] = vz0 + i*dv;   // vertex-center
            for (int i=0;i<nz;  i++)	Z[i]  =  z0 + (i+0.5)*dz;

            printf("\n\nNuOsc with max OpenMP core: %d\n\n", omp_get_max_threads() );
            printf("   Domain: vz:( %12f %12f )  nvz = %5d                     dv = %g\n", vz0,vz1, nvz, dv);
            printf("            z:( %12f %12f )  nz  = %5d    buffer zone =%2d  dz = %g\n", z0,z1, nz, gz, dz);
            printf("   dt = %g     CFL = %g\n", dt, CFL);
#ifdef BC_PERI
            printf("   Use Periodic boundary\n");
#else
            printf("   Use open boundary\n");
#endif
#ifdef KO_ORD_3
            printf("   Use 3-th order KO dissipation\n");
#else
            printf("   Use 5-th order KO dissipation\n");
#endif
            printf("   KO eps = %g\n", ko);

            //anafile.open("rate.dat", std::ofstream::out | std::ofstream::trunc);
            //if(!anafile) cout << "*** Open fails: " << "./rate.dat" << endl;

            // dump at highest/lowest v-mode
            ee_vl.open("ee_vl.dat", std::ofstream::out | std::ofstream::trunc);
            ee_vl << nz << " " << z0 << " " << z1 << endl;
            ee_vh.open("ee_vh.dat", std::ofstream::out | std::ofstream::trunc);
            ee_vh << nz << " " << z0 << " " << z1 << endl;
            exr_vl.open("exr_vl.dat", std::ofstream::out | std::ofstream::trunc);
            exr_vl << nz << " " << z0 << " " << z1 << endl;
            exr_vh.open("exr_vh.dat", std::ofstream::out | std::ofstream::trunc);
            exr_vh << nz << " " << z0 << " " << z1 << endl;
            //exi_vl.open("exi_vl.dat", std::ofstream::out | std::ofstream::trunc);
            //exi_vl << nz << " " << z0 << " " << z1 << endl;
            //exi_vh.open("exi_vh.dat", std::ofstream::out | std::ofstream::trunc);
            //exi_vh << nz << " " << z0 << " " << z1 << endl;
            ee_vm.open("ee_vm.dat", std::ofstream::out | std::ofstream::trunc);
            ee_vm << nz << " " << z0 << " " << z1 << endl;
            con_vl.open("con_vl.dat", std::ofstream::out | std::ofstream::trunc);
            con_vl << nz << " " << z0 << " " << z1 << endl;
            con_vh.open("con_vh.dat", std::ofstream::out | std::ofstream::trunc);
            con_vh << nz << " " << z0 << " " << z1 << endl;
            con_vm.open("con_vm.dat", std::ofstream::out | std::ofstream::trunc);
            con_vm << nz << " " << z0 << " " << z1 << endl;
        }

        //NuOsc(const NuOsc &v) {  // Copy constructor to be checked.
        //    NuOsc(v.nz, v.nvz);
        //}

        ~NuOsc() {
            delete[] vz;
            delete[] Z;
            delete[] con1;
            delete v_stat, v_rhs, v_pre, v_cor;

            anafile.close();

            ee_vl.close();    ee_vh.close();    ee_vm.close();
            exr_vl.close();  exr_vh.close();
            exi_vl.close();  exi_vh.close();
            con_vl.close();  con_vh.close();  con_vm.close();
        }

        void set_mu(real mu_) { 
            mu = mu_;
            printf("   Setting mu = %f\n\n", mu);
        }

        void fillInitValue(real f0, real alpha, real beta);
        void updatePeriodicBoundary(FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);
        void step_rk4();
        void calRHS(FieldVar* out, const FieldVar * in);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
        void analysis();
        void _analysis_v(real res[], const real var[]);
        void _analysis_c(real res[], const real vr[], const real vi[]);
        void angle_integrated(real &res, const real vr[], const real vi[]);
        void dumpv(const real v[]);
        void write_z_at_vz();
        void eval_conserved();
        void write_bin(const int t);

        inline void _out_tmp(real res[], const real v[]) {
            res[0] = v[idx(0,             nz/2)];
            res[1] = v[idx(int(nvz*0.75), nz/2)];
            res[2] = v[idx(nvz-1,         nz/2)];
        }

};

void NuOsc::eval_conserved() {

    real p1, p2, p3;
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {

            p1 = v_stat->ex_re[idx(i,j)];
            p2 = v_stat->ex_im[idx(i,j)];
            p3 = 2*v_stat->ee[idx(i,j)] - 1.0;
            con1[idx(i,j)] = p1*p1 + p2*p2 + 0.25*p3*p3;
        }
    }
}

void NuOsc::fillInitValue(real f0, real alpha, real beta = 0.0) {

#pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {
            v_stat->ee   [idx(i,j)] = g(vz[i], 1.0, 0.6, 7.608447e-01)*0.5 * (1.0+eps_(Z[j], 0.0));
            v_stat->ex_re[idx(i,j)] = g(vz[i], 1.0, 0.6, 7.608447e-01)*0.5 *      eps (Z[j], 0.0);
            v_stat->ex_im[idx(i,j)] = 0.0;
        }
    }
#pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {
            v_stat->bee   [idx(i,j)] = 0.97*g(vz[i], 1.0, 0.53, 6.736495e-01)*0.5 * (1.0+eps_(Z[j], 0.0));
            v_stat->bex_re[idx(i,j)] = 0.97*g(vz[i], 1.0, 0.53, 6.736495e-01)*0.5 *      eps (Z[j], 0.0);
            v_stat->bex_im[idx(i,j)] = 0.0;
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
    outfile.write((char *) v->ex_re,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->ex_im,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bee,    (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bex_re, (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) v->bex_im, (nz+2*gz)*nvz*sizeof(real));
    outfile.close();
    printf("		Write %d x %d into %s\n", nvz, nz+2*gz, filename);
}

void NuOsc::updatePeriodicBoundary(FieldVar * in) {

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<gz; j++) {
            //lower side
            in->ee    [idx(i,-j-1)] = in->ee    [idx(i,nz-j-1)];
            in->ex_re [idx(i,-j-1)] = in->ex_re [idx(i,nz-j-1)];
            in->ex_im [idx(i,-j-1)] = in->ex_im [idx(i,nz-j-1)];
            in->bee   [idx(i,-j-1)] = in->bee   [idx(i,nz-j-1)];
            in->bex_re[idx(i,-j-1)] = in->bex_re[idx(i,nz-j-1)];
            in->bex_im[idx(i,-j-1)] = in->bex_im[idx(i,nz-j-1)];
            //upper side
            in->ee    [idx(i,nz+j)] = in->ee    [idx(i,j)];
            in->ex_re [idx(i,nz+j)] = in->ex_re [idx(i,j)];
            in->ex_im [idx(i,nz+j)] = in->ex_im [idx(i,j)];
            in->bee   [idx(i,nz+j)] = in->bee   [idx(i,j)];
            in->bex_re[idx(i,nz+j)] = in->bex_re[idx(i,j)];
            in->bex_im[idx(i,nz+j)] = in->bex_im[idx(i,j)];
        }
}

void NuOsc::updateInjetOpenBoundary(FieldVar * in) {
    // Cell-center for z:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]
    // Open boundary at two ends
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
	for (int j=0;j<gz; j++) {
        //lower open side, estimated simply by extrapolation
        in->ee    [idx(i,-j-1)] = in->ee    [idx(i,-j)]*2 - in->ee    [idx(i,-j+1)];
        in->ex_re [idx(i,-j-1)] = in->ex_re [idx(i,-j)]*2 - in->ex_re [idx(i,-j+1)];
        in->ex_im [idx(i,-j-1)] = in->ex_im [idx(i,-j)]*2 - in->ex_im [idx(i,-j+1)];
        in->bee   [idx(i,-j-1)] = in->bee   [idx(i,-j)]*2 - in->bee   [idx(i,-j+1)];
        in->bex_re[idx(i,-j-1)] = in->bex_re[idx(i,-j)]*2 - in->bex_re[idx(i,-j+1)];
        in->bex_im[idx(i,-j-1)] = in->bex_im[idx(i,-j)]*2 - in->bex_im[idx(i,-j+1)];

        //upper open side, estimated simply by extrapolation
        in->ee    [idx(i,nz+j)] = in->ee    [idx(i,nz+j-1)]*2 - in->ee    [idx(i,nz+j-2)];
        in->ex_re [idx(i,nz+j)] = in->ex_re [idx(i,nz+j-1)]*2 - in->ex_re [idx(i,nz+j-2)];
        in->ex_im [idx(i,nz+j)] = in->ex_im [idx(i,nz+j-1)]*2 - in->ex_im [idx(i,nz+j-2)];
        in->bee   [idx(i,nz+j)] = in->bee   [idx(i,nz+j-1)]*2 - in->bee   [idx(i,nz+j-2)];
        in->bex_re[idx(i,nz+j)] = in->bex_re[idx(i,nz+j-1)]*2 - in->bex_re[idx(i,nz+j-2)];
        in->bex_im[idx(i,nz+j)] = in->bex_im[idx(i,nz+j-1)]*2 - in->bex_im[idx(i,nz+j-2)];
	}
}

void NuOsc::calRHS(FieldVar * out, const FieldVar * in) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {

            real *ee    = &(in->ee    [idx(i,j)]);
            real *exr   = &(in->ex_re [idx(i,j)]);
            real *exi   = &(in->ex_im [idx(i,j)]);
            real *bee   = &(in->bee   [idx(i,j)]);
            real *bexr  = &(in->bex_re[idx(i,j)]);
            real *bexi  = &(in->bex_im[idx(i,j)]);

            // 1) prepare terms for -i [H0, rho]
            out->ee    [idx(i,j)] = -pmo *   2*st*exi [0];
            out->ex_re [idx(i,j)] = -pmo *   2*ct*exi [0];
            out->ex_im [idx(i,j)] =  pmo * ( 2*ct*exr [0] + st*( 2*ee[0] -1 ) );
            out->bee   [idx(i,j)] = -pmo *   2*st*bexi[0];
            out->bex_re[idx(i,j)] = -pmo *   2*ct*bexi[0];
            out->bex_im[idx(i,j)] =  pmo * ( 2*ct*bexr[0] + st*( 2*bee[0]-1 ) );

#ifndef ADVEC_OFF
#if 1
            // advection term: (lopsided finite differencing)
            int sv = sgn(vz[i]);
            real factor = sv*vz[i]/(12*dz);
            #define ADV_FD(x)  ( x[-3*sv] - 6*x[-2*sv] + 18*x[-sv] - 10*x[0] - 3*x[sv] )
#else
            // 2) advection term: (central FD)
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor = -vz[i]/(12*dz);
            #define ADV_FD(x)  (  (x[-2]-x[2]) - 8.0*(x[-1]-x[1]) )
#endif
            out->ee    [idx(i,j)] += factor * ADV_FD(ee);
            out->ex_re [idx(i,j)] += factor * ADV_FD(exr);
            out->ex_im [idx(i,j)] += factor * ADV_FD(exi);
            out->bee   [idx(i,j)] += factor * ADV_FD(bee);
            out->bex_re[idx(i,j)] += factor * ADV_FD(bexr);
            out->bex_im[idx(i,j)] += factor * ADV_FD(bexi);
	    #undef ADV_FD
#endif

    if (mu>0.0) {
            // 3) interaction terms: vz-integral with a simple trapezoidal rule (can be optimized later)
            real Iee    = 0;
            real Iexr   = 0;
            real Iexi   = 0;
            real Ibee   = 0;
            real Ibexr  = 0;
            real Ibexi  = 0;
            for (int k=1;k<nvz-1; k++) {   // vz' integral
                real eep    = (in->ee    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   += 2*mu* (1-vz[i]*vz[k])*  (       exr[0] *(expi + bexpi) -    exi[0]*(expr- bexpr) );
                Iexr  +=   mu* (1-vz[i]*vz[k])*  (   (1-2*ee[0])*(expi + bexpi) +  2*exi[0]*(eep - beep ) );
                Iexi  +=   mu* (1-vz[i]*vz[k])*  (  -(1-2*ee[0])*(expr - bexpr) -  2*exr[0]*(eep - beep ) );
                Ibee  += 2*mu* (1-vz[i]*vz[k])*  (      bexr[0] *(expi + bexpi) +   bexi[0]*(expr- bexpr) );
                Ibexr +=   mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expi + bexpi) - 2*bexi[0]*(eep - beep ) );
                Ibexi +=   mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expr - bexpr) + 2*bexr[0]*(eep - beep ) );
            }
            {   int k=0;  // Deal with end point for integral in vertex-center grid
                real eep    = (in->ee    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=     mu* (1-vz[i]*vz[k])*  (       exr[0] *(expi + bexpi) -    exi[0]*(expr- bexpr) );
                Iexr  += 0.5*mu* (1-vz[i]*vz[k])*  (   (1-2*ee[0])*(expi + bexpi) +  2*exi[0]*(eep - beep ) );
                Iexi  += 0.5*mu* (1-vz[i]*vz[k])*  (  -(1-2*ee[0])*(expr - bexpr) -  2*exr[0]*(eep - beep ) );
                Ibee  +=     mu* (1-vz[i]*vz[k])*  (      bexr[0] *(expi + bexpi) +   bexi[0]*(expr- bexpr) );
                Ibexr += 0.5*mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expi + bexpi) - 2*bexi[0]*(eep - beep ) );
                Ibexi += 0.5*mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expr - bexpr) + 2*bexr[0]*(eep - beep ) );
            }
            {   int k=nvz-1;  // Deal with end point for integral in vertex-center grid
                real eep    = (in->ee    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=     mu* (1-vz[i]*vz[k])*  (       exr[0] *(expi + bexpi) -    exi[0]*(expr- bexpr) );
                Iexr  += 0.5*mu* (1-vz[i]*vz[k])*  (   (1-2*ee[0])*(expi + bexpi) +  2*exi[0]*(eep - beep ) );
                Iexi  += 0.5*mu* (1-vz[i]*vz[k])*  (  -(1-2*ee[0])*(expr - bexpr) -  2*exr[0]*(eep - beep ) );
                Ibee  +=     mu* (1-vz[i]*vz[k])*  (      bexr[0] *(expi + bexpi) +   bexi[0]*(expr- bexpr) );
                Ibexr += 0.5*mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expi + bexpi) - 2*bexi[0]*(eep - beep ) );
                Ibexi += 0.5*mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expr - bexpr) + 2*bexr[0]*(eep - beep ) );
            }

            // 3.1) calculate integral with simple trapezoidal rule
            out->ee    [idx(i,j)] += dv*Iee;
            out->ex_re [idx(i,j)] += dv*Iexr;
            out->ex_im [idx(i,j)] += dv*Iexi;
            out->bee   [idx(i,j)] += dv*Ibee;
            out->bex_re[idx(i,j)] += dv*Ibexr;
            out->bex_im[idx(i,j)] += dv*Ibexi;
    } // end of mu-part

#ifdef KO_ORD_3
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps = -ko/dz/16.0;
            #define KO_FD(x)  ( x[-2] + x[2] - 4*(x[-1]+x[1]) + 6*x[0] )
#else
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps = -ko/dz/64.0;
            #define KO_FD(x)  ( x[-3] + x[3] - 6*(x[-2]+x[2]) + 15*(x[-1]+x[1]) - 20*x[0] )
#endif
            out->ee    [idx(i,j)] += ko_eps * KO_FD(ee);
            out->ex_re [idx(i,j)] += ko_eps * KO_FD(exr);
            out->ex_im [idx(i,j)] += ko_eps * KO_FD(exi);
            out->bee   [idx(i,j)] += ko_eps * KO_FD(bee);
            out->bex_re[idx(i,j)] += ko_eps * KO_FD(bexr);
            out->bex_im[idx(i,j)] += ko_eps * KO_FD(bexi);
            #undef KO_FD
        }
}

// v0 = v1 + a * v2
void NuOsc::vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            int k = idx(i,j);
            v0->ee    [k] = v1->ee    [k] + a * v2->ee    [k];
            v0->ex_re [k] = v1->ex_re [k] + a * v2->ex_re [k];
            v0->ex_im [k] = v1->ex_im [k] + a * v2->ex_im [k];
            v0->bee   [k] = v1->bee   [k] + a * v2->bee   [k];
            v0->bex_re[k] = v1->bex_re[k] + a * v2->bex_re[k];
            v0->bex_im[k] = v1->bex_im[k] + a * v2->bex_im[k];
        }
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            int k = idx(i,j);
            v0->ee    [k] = v1->ee    [k] + a * (v2->ee    [k] + v3->ee    [k]);
            v0->ex_re [k] = v1->ex_re [k] + a * (v2->ex_re [k] + v3->ex_re [k]);
            v0->ex_im [k] = v1->ex_im [k] + a * (v2->ex_im [k] + v3->ex_im [k]);
            v0->bee   [k] = v1->bee   [k] + a * (v2->bee   [k] + v3->bee   [k]);
            v0->bex_re[k] = v1->bex_re[k] + a * (v2->bex_re[k] + v3->bex_re[k]);
            v0->bex_im[k] = v1->bex_im[k] + a * (v2->bex_im[k] + v3->bex_im[k]);
        }
}

void NuOsc::step_rk4() {
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

    phy_time += dt;
}

void NuOsc::_analysis_v(real res[], const real var[]) {

    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;

#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax)
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            real val = var[idx(i,j)];
            if (val>vmax) vmax = val;
            if (val<vmin) vmin = val;
            sum  += val;
            sum2 += val*val;
        }

    // min, max, avg, std
    res[0] = vmin;
    res[1] = vmax;
    res[2] = sum/(nz*nvz);
    res[3] = sqrt( sum2/(nz*nvz) - res[2]*res[2]  );
}

void NuOsc::_analysis_c(real res[], const real vr[], const real vi[]) {

    real vmin =  1.e32;
    real vmax = -1.e32;
    real sum  = 0;
    real sum2 = 0;

#pragma omp parallel for reduction(+: sum) reduction(+:sum2) reduction(min:vmin) reduction(max:vmax)
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            int ij = idx(i,j);
            real val = vr[ij]*vr[ij]+vi[ij]*vi[ij];
            if (val>vmax) vmax = val;
            if (val<vmin) vmin = val;
            sum  += val;
            sum2 += val*val;
        }

    // min, max, avg, std
    res[0] = vmin;
    res[1] = vmax;
    res[2] = sum/(nz*nvz);
    res[3] = sqrt( sum2/(nz*nvz) - res[2]*res[2]  );
}

/* return the angle-integrated |v| == sqrt(vr**2+vi**2) */
void NuOsc::angle_integrated(real &res, const real vr[], const real vi[]) {
    int loc = nz/2;
    real sum = 0.0;
#pragma omp parallel for reduction(+: sum)
    for (int k=1;k<nvz-1; k++) {   // vz' integral
        sum   += (vr[idx(loc,k)]*vr[idx(loc,k)]+vi[idx(loc,k)]*vi[idx(loc,k)]);
    }
    sum += 0.5*(vr[idx(loc,0)]*vr[idx(loc,0)]+vi[idx(loc,0)]*vi[idx(loc,0)]) +
        0.5*(vr[idx(loc,nvz-1)]*vr[idx(loc,nvz-1)]+vi[idx(loc,nvz-1)]*vi[idx(loc,nvz-1)]);
    res = dv*sum;
}

/* return the angle-integrated |v| == sqrt(vr**2+vi**2) */
//void NuOsc::angle_integrated(real &res, const real vr[], const real vi[]) {
//    int loc = nz/2;
//    real sum = 0;
//    #pragma omp parallel for reduction(+: sum)
//    for (int k=0;k<nvz; k++) {   // vz' integral
//	sum   += dv*sqrt(vr[idx(loc,k)]*vr[idx(loc,k)]+vi[idx(loc,k)]*vi[idx(loc,k)]);
//    }
//    res = sum;
//}/

void NuOsc::analysis() {

    eval_conserved();

    //real statis1[4], statis2[4];
    //_analysis_c(statis1, v_stat-> ex_re, v_stat-> ex_im);
    //_analysis_v(statis2, v_stat->bee);

    //printf("Time: %.5f  min/max/std of |ex|:( %9.2g %9.2g %9.2g ) |bee|:( %9.2g %9.2g %9.2g)\n", phy_time,
    //			statis1[0], statis1[1], statis1[3],
    //			statis2[0], statis2[1], statis2[3]);
    //anafile << phy_time <<" "<< statis1[0]<< " " << statis1[1] << " " << statis1[3] << " "
    //                         << statis2[0]<< " " << statis2[1] << " " << statis2[3] << " "
    //                         << probe0 << endl;

    /*
       real pee [4];  _out_tmp(pee, v_stat->ee);
       real pexr[4];  _out_tmp(pexr, v_stat->ex_re);
       real pexi[4];  _out_tmp(pexi, v_stat->ex_im);
       real ai;
       angle_integrated(ai, v_stat->ex_re,  v_stat->ex_im);
       printf("Time: %.5f  |ee|: %9.2g %9.2g %9.2g  |exr|: %9.2g %9.2g %9.2g   |exi|: %9.2g %9.2g %9.2g  A= %g\n", phy_time,
       pee[0],  pee[1],  pee[2],
       pexr[0], pexr[1], pexr[2],
       pexi[0], pexi[1], pexi[2], ai );
       anafile << phy_time <<" "<< pee[0]<<  " " << pee[1] <<  " " << pee[2] << " "
       << pexr[0]<< " " << pexr[1] << " " << pexr[2] << " "
       << pexi[0]<< " " << pexi[1] << " " << pexi[2] << " " << ai << endl;
    */

    real st[4];  _analysis_v(st, con1);

    printf("Phy time: %.5f  min/max/std of |sum p_i^2|: %9.2g %9.2g %9.2g\n", phy_time, st[0], st[1], st[3]);
}

void NuOsc::dumpv(const real v[]) {
    cout << "	=== ee component ===" << endl;
    for (int i=0;i<nvz; i++) {
        for (int j=-gz;j<nz+gz; j++) {
            printf("%10.2e ", v[idx(i,j)]);
        }
        cout << endl;
    }
    cout << endl;
}

void NuOsc::write_z_at_vz() {
    FieldVar *v = v_stat;

#define WRITE_Z_AT(HANDLE, VAR, V_IDX) \
        HANDLE << phy_time << " "; \
        for (int i=0;i<nz; i++) HANDLE << std::setprecision(14) << VAR[idx(V_IDX, i)] << " "; \
        HANDLE << endl;
    // f(z) at the lowest  v-mode
    WRITE_Z_AT(ee_vl,  v->ee,    0)
    WRITE_Z_AT(exr_vl, v->ex_re, 0)
    //WRITE_Z_AT(exi_vl, v->ex_im, 0)
    // f(z) at the highest v-mode
    WRITE_Z_AT(ee_vh,  v->ee,    nvz-1)
    WRITE_Z_AT(exr_vh, v->ex_re, nvz-1)
    //WRITE_Z_AT(exi_vh, v->ex_im, nvz-1)

    // f(z) at the v=0
    WRITE_Z_AT(ee_vm,  v->ee,    int((nvz-1)/2))

    WRITE_Z_AT(con_vl, con1,    0)
    WRITE_Z_AT(con_vm, con1,    int((nvz-1)/2))
    WRITE_Z_AT(con_vh, con1,    nvz-1)
#undef WRITE_Z_AT

}

int main(int argc, char *argv[]) {

    real dz  = 0.125;
    real z0  = -100;      real z1  =  -z0;
    real vz0 = -1;       real vz1 =  -vz0;    int nvz = 8 + 1;
    real cfl = 0.25;     real ko = 1e-4;

    real mu  = 0.0;

    // Parse input argument
    for (int t = 1; argv[t] != 0; t++) {
        if (strcmp(argv[t], "--dz") == 0 )  {
            dz  = atof(argv[t+1]);     t+=1;
        } else if (strcmp(argv[t], "--zmax") == 0 )  {
            z1   = atof(argv[t+1]);    t+=1;
            z0   = -z1;
        } else if (strcmp(argv[t], "--cfl") == 0 )  {
            cfl   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--nvz") == 0 )  {
            nvz   = atoi(argv[t+1]);    t+=1;
            assert(nvz%2==1);
        } else if (strcmp(argv[t], "--ko") == 0 )  {
            ko    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--mu") == 0 )  {
            mu    = atof(argv[t+1]);    t+=1;
        } else {
            printf("Unreconganized parameters %s!\n", argv[t]);
            exit(0);
        }
    }
    int nz  = int((z1-z0)/dz);
    
#if 0
    // check ~10 advection cycle to see if return to original state (for PBC)
    int ANAL_EVERY = nz/cfl * 1 ;
    int END_STEP   = ANAL_EVERY * 10;
    int DUMP_EVERY = 99999999;
#else
    // check every 50-step before hitting boundary
    int END_STEP   = int(nz/cfl*0.55);
    int ANAL_EVERY = END_STEP / 50;
    int DUMP_EVERY = ANAL_EVERY*9999;
#endif

    // Initialize simuation
    NuOsc state(nvz, nz, vz0, vz1, z0, z1, cfl, ko);
    state.set_mu(mu);

    // initial value
    real f0    = 1.0;
    real alpha = 0.97;
    state.fillInitValue(f0, alpha);

    // analysis for t=0
    state.analysis();
    state.write_z_at_vz();
    //state.write_bin(0);

    for (int t=1; t<=END_STEP; t++) {
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            state.analysis();
            state.write_z_at_vz();
        }
        if ( t%DUMP_EVERY==0) {
            state.write_bin(t);
        }
    }

    printf("Completed.\n");
    return 0;
}
