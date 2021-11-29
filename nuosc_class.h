#pragma once

#ifdef NVTX
#include <nvToolsExt.h>
#endif
#ifdef OPENACC
#include <openacc.h>
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#ifdef PAPI
#include <papi.h>
#endif

#include <algorithm>
#include <list>

#define BC_PERI
#define KO_ORD_3
#define ADVEC_CENTER_FD
//#define ADVEC_UPWIND  ## always blow-up
//#define ADVEC_OFF

//#define CELL_CENTER_V

using std::cout;
using std::endl;
using std::cin;

typedef double real;

typedef struct Vars {
    real* ee;
    real* xx;
    real* ex_re;
    real* ex_im;
    real* bee;
    real* bxx;
    real* bex_re;
    real* bex_im;

    Vars(int size) {
        ee     = new real[size](); // all init to zero
        xx     = new real[size]();
        ex_re  = new real[size]();
        ex_im  = new real[size]();
        bee    = new real[size]();
        bxx    = new real[size]();
        bex_re = new real[size]();
        bex_im = new real[size]();
#pragma acc enter data create(this,ee[0:size],xx[0:size],ex_re[0:size],ex_im[0:size],bee[0:size],bxx[0:size],bex_re[0:size],bex_im[0:size])
    }
    ~Vars() {
#pragma acc exit data delete(ee, xx, ex_re, ex_im, bee, bxx, bex_re, bex_im, this)
        delete[] ee;
        delete[] xx;
        delete[] ex_re;
        delete[] ex_im;
        delete[] bee;
        delete[] bxx;
        delete[] bex_re;
        delete[] bex_im;
    }
} FieldVar;

typedef struct stat {
    real min;
    real max;
    real sum;
    real avg;
    real std;
} FieldStat;



/*
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
*/


class NuOsc {
    public:
        real phy_time, dt;

        // all coordinates
        real  *vz, *vw, *Z;
        // all field variables
        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;
        real *P1,  *P2,  *P3,  *relN,  *relP;
        real *P1b, *P2b, *P3b, *relNb, *relPb;
        real *G0,*G0b;

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
        real mu = 1.0;      // can be set by set_mu()
        bool renorm = false;  // can be set by set_renorm()

        std::ofstream anafile;

        // text output of f(z) at certain v-mode
        std::ofstream ee_vl,  ee_vh, ee_vm;
        std::ofstream p1_vm, p2_vm, p3_vm;
        std::ofstream p1_v, p2_v, p3_v;
        std::ofstream ogv, ogvb;

        inline unsigned int idx(const int i, const int j) { return (j+gz)*nvz + i; }    // i:vz   j:z (last index)
        inline unsigned int idz(const int j) { return j; }

        NuOsc(const int  nvz_, const int   nz_,
                const real vz0_, const real vz1_,
                const real  z0_, const real  z1_,
                const real CFL_, const real  ko_, const int gz_ = 2) : phy_time(0.), ko(ko_)  {

            vz0 = vz0_;  vz1 = vz1_;
            z0  = z0_;    z1 = z1_;
            CFL = CFL_;

            nvz = nvz_;
            nz  = nz_;
            gz  = gz_;
            int size = (nz+2*gz)*(nvz);

            vz     = new real[nvz];
            vw     = new real[nvz];   // quadrature of v
            Z      = new real[nz];
            G0     = new real[size];
            G0b    = new real[size];
            P1    = new real[size];
            P2    = new real[size];
            P3    = new real[size];
            P1b   = new real[size];
            P2b   = new real[size];
            P3b   = new real[size];
            relP  = new real[size];
            relN  = new real[size];
            relPb = new real[size];
            relNb = new real[size];

            v_stat = new FieldVar(size);
            v_rhs  = new FieldVar(size);
            v_pre  = new FieldVar(size);
            v_cor  = new FieldVar(size);
#pragma acc enter data create(v_stat[0:1], v_rhs[0:1], v_pre[0:1], v_cor[0:1]) attach(v_stat, v_rhs, v_pre, v_cor)
            dz = (z1-z0)/nz;       // cell-center
            dt = dz*CFL;
            for (int i=0;i<nz;  i++)	Z[i]  =  z0 + (i+0.5)*dz;
#ifdef CELL_CENTER_V
            dv = (vz1-vz0)/nvz;      // for v as cell-center
            for (int i=0;i<nvz; i++)	{
                vz[i] = vz0 + (i+0.5)*dv;   // cell-center
                vw[i] = 1.0;
            }
#else
            dv = (vz1-vz0)/(nvz-1);      // for v as vertex-center
            for (int i=0;i<nvz; i++) {
                vz[i] = vz0 + i*dv;   // vertex-center
                vw[i] = (i==0 || i==nvz-1)? 0.5 : 1.0;
            }
#endif

            printf("\n\nNuOsc with max OpenMP core: %d\n\n", omp_get_max_threads() );
            printf("   Domain: vz:( %12f %12f )  nvz = %5d                     dv = %g\n", vz0,vz1, nvz, dv);
            printf("            z:( %12f %12f )  nz  = %5d    buffer zone =%2d  dz = %g\n", z0,z1, nz, gz, dz);
            printf("   dt = %g     CFL = %g\n", dt, CFL);
#ifdef BC_PERI
            printf("   Use Periodic boundary\n");
#else
            printf("   Use open boundary\n");
#endif
#ifndef KO_ORD_3
            printf("   Use 5-th order KO dissipation\n");
#else
            printf("   Use 3-th order KO dissipation\n");
#endif
            printf("   KO eps = %g\n", ko);

#ifndef ADVEC_OFF
#if defined(ADVEC_CENTER_FD)
            printf("   Use center-FD for advaction\n");
#elif defined(ADVEC_UPWIND)
            printf("   Use upwinded for advaction. (EXP. Always blowup!!\n");
#else
            printf("   Use lopsided FD for advaction\n");
#endif
#endif

            anafile.open("analysis.dat", std::ofstream::out | std::ofstream::trunc);
            if(!anafile) cout << "*** Open fails: " << "./analysis.dat" << endl;

            // dump f(z) at certain v-mode
            p1_vm.open("p1_vm.dat", std::ofstream::out | std::ofstream::trunc);
            p1_vm << nz << " " << z0 << " " << z1 << endl;
            p2_vm.open("p2_vm.dat", std::ofstream::out | std::ofstream::trunc);
            p2_vm << nz << " " << z0 << " " << z1 << endl;
            p3_vm.open("p3_vm.dat", std::ofstream::out | std::ofstream::trunc);
            p3_vm << nz << " " << z0 << " " << z1 << endl;
            p1_v.open("p1_v.dat", std::ofstream::out | std::ofstream::trunc);
            p1_v << nz << " " << z0 << " " << z1 << endl;
            p2_v.open("p2_v.dat", std::ofstream::out | std::ofstream::trunc);
            p2_v << nz << " " << z0 << " " << z1 << endl;
            p3_v.open("p3_v.dat", std::ofstream::out | std::ofstream::trunc);
            p3_v << nz << " " << z0 << " " << z1 << endl;

            ogv.open("ogv.dat", std::ofstream::out | std::ofstream::trunc);
            ogv << nvz << " " << vz0 << " " << vz1 << endl;
            ogvb.open("ogvb.dat", std::ofstream::out | std::ofstream::trunc);
            ogvb << nvz << " " << vz0 << " " << vz1 << endl;

        }

        //NuOsc(const NuOsc &v) {  // Copy constructor to be checked.
        //    NuOsc(v.nz, v.nvz);
        //}


        ~NuOsc() {
            delete[] vz;  delete[] Z;
            delete[] G0;
            delete[] G0b;
            delete[] P1;  delete[] P2;  delete[] P3;  delete[] relP;  delete[] relN;
            delete[] P1b; delete[] P2b; delete[] P3b; delete[] relPb; delete[] relNb;
#pragma acc exit data delete(v_stat, v_rhs, v_pre, v_cor)
            delete v_stat, v_rhs, v_pre, v_cor;


            anafile.close();

            ee_vl.close();    ee_vh.close();    ee_vm.close();
            p1_vm.close();  p2_vm.close();    p3_vm.close();
            p1_v.close();   p2_v.close();      p3_v.close();
            ogv.close();    ogvb.close();

        }

        void set_mu(real mu_) {
            mu = mu_;
            printf("   Setting mu = %f\n", mu);
        }

        void set_renorm(bool renorm_) {
            renorm = renorm_;
            printf("   Setting renorm = %d\n", renorm);
        }

        void fillInitValue(real f0, real alpha, real lnue, real lnueb, int ipt, real eps0, real lzpt);

        void updatePeriodicBoundary (FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);
        void step_rk4();
        void calRHS(FieldVar* out, const FieldVar * in);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
        void analysis();
        FieldStat _analysis_v(const real var[]);
        FieldStat _analysis_c(const real vr[], const real vi[]);
        void angle_integrated(real &res, const real vr[], const real vi[]);
        void eval_conserved(const FieldVar* v0);
        void renormalize(const FieldVar* v0);

        void write_fz();
        void write_fv();
        void write_bin(const int t);
        void output_detail(const char* fn);

};
