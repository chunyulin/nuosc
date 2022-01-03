#pragma once

//==== Major configuration macros to use  =========
//#define COSENU2D
#define BC_PERI
#define KO_ORD_3
#define VACUUM_OFF

//#define ADVEC_OFF
//#define ADVEC_UPWIND    ## always blow-up
//#define ADVEC_LOPSIDED  ## seems not good neither
//=================================================

#include <omp.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <list>
#include <vector>
#include <algorithm>

#ifdef PAPI
#include <papi.h>
#endif

#ifdef NVTX
#include <nvToolsExt.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::vector;

#ifdef COSENU2D
#define COLLAPSE_LOOP 3

#define PARFORALL(i,j,v) \
    _Pragma("omp parallel for collapse(3)") \
    _Pragma("acc parallel loop collapse(3)") \
    for (int i=0;i<ny; i++) \
    for (int j=0;j<nz; j++) \
    for (int v=0;v<nv; v++)

#define FORALL(i,j,v) \
    for (int i=0;i<ny; i++) \
    for (int j=0;j<nz; j++) \
    for (int v=0;v<nv; v++)

#else
#define COLLAPSE_LOOP 2

#define PARFORALL(i,j,v) \
    for (int i=0;i<1; i++) \
    _Pragma("omp parallel for collapse(2)") \
    _Pragma("acc parallel loop collapse(2)") \
    for (int j=0;j<nz; j++) \
    for (int v=0;v<nv; v++)

#define FORALL(i,j,v) \
    for (int i=0;i<1; i++) \
    for (int j=0;j<nz; j++) \
    for (int v=0;v<nv; v++)

#endif

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


// Each "SkimShot" is basically a binary file per iteration 
// that contains reduced snapshot of multiple vaiables.
// I choose noe to combine multiple iteration in a file, as each iteration is large for nD-simulation.  
typedef struct SkimShot_struct {
    std::list<real*> var_list;
    string fntpl;
    int every;
    int sy, sz, sv;         // reduced grid in each dimension
    vector<int> y_slices;   // coordinate for the reduced dimension
    vector<int> v_slices;

    // init with specified v-coordinate...    
    SkimShot_struct(std::list<real*> var_list_, string fntpl_, uint every_, uint sy_, uint sz_, vector<int> v_slices_) {
        var_list = var_list_;
        fntpl = fntpl_;
        every = every_;
        sy = sy_;
        sz = sz_;
        v_slices = v_slices_;
    }
    // init with specified y and v-coordinate...    
    SkimShot_struct(std::list<real*> var_list_, string fntpl_, uint every_, vector<int> y_slices_, uint sz_, vector<int> v_slices_) {
        var_list = var_list_;
        fntpl = fntpl_;
        every = every_;
        y_slices = y_slices_;
        v_slices = v_slices_;
        sz = sz_;
    }
} SkimShot;


inline void swap(FieldVar **a, FieldVar **b) { FieldVar *tmp = *a; *a = *b; *b = tmp; }
inline real random_amp(real a) { return a * rand() / RAND_MAX; }
template <typename T> int sgn(T val) {    return (T(0) < val) - (val < T(0));   }


int gen_v2d_simple(const int nv_, real *& vw, real *& vy, real *& vz);
int gen_v1d_vertex_center(const int nv_, real *& vw, real *& vz);
int gen_v1d_cell_center(const int nv_, real *& vw, real *& vz);

vector<int> gen_skimmed_vslice_index(uint nv_target, uint nv_in, const real * vy, const real * vz);
vector<int> gen_skimmed_vslice_index(uint nv_target, uint nv_in);


class NuOsc {
    public:
        real phy_time, dt;

        real  *vw;                 // integral quadrature
        int nv;       // # of v cubature points.

        // z-coordinate
        real  *vz, *Z;
        real z0,  z1, dz;
        int nz, gz;  // Dim of z  (the last dimension, the one with derivatives. Cell-center grid used.)

        // y-coordinate
        real  *vy, *Y;
        real y0,  y1, dy;
        int ny, gy;  // Dim of y

        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;  // field variables

        real *P1,  *P2,  *P3,  *dN,  *dP;
        real *P1b, *P2b, *P3b, *dNb, *dPb;
        real *G0,*G0b;

        real CFL;
        real ko;

        const real theta = 1e-6;
        const real ct = cos(2*theta);
        const real st = sin(2*theta);
        const int  pmo = 0; // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
        real mu = 1.0;      // can be set by set_mu()
        bool renorm = false;  // can be set by set_renorm()

        std::ofstream anafile;
        std::list<SkimShot> skimshots;
        

#ifdef COSENU2D
        inline unsigned long idx(const int i, const int j, const int v) { return   ( (i+gy)*(nz+2*gz) + j+gz)*nv + v; }    //  i:y j:z
#else
        inline unsigned long idx(const int i, const int j, const int v) { return   (j+gz)*nv + v; }
#endif

        NuOsc(const int  nv_, 
                const int   ny_, const int   nz_, 
                const real  y0_, const real  y1_, const real  z0_, const real  z1_,
                const real CFL_, const real  ko_) : phy_time(0.), ko(ko_)  {

            // coordinates~~

            nz  = nz_;  z0  = z0_;  z1 = z1_;
            Z      = new real[nz];
            dz = (z1-z0)/nz;       // cell-center
            for (int i=0;i<nz;  i++)	Z[i]  =  z0 + (i+0.5)*dz;
#ifndef KO_ORD_3
            gz  = 3;
#else
            gz  = 2;
#endif

#ifdef COSENU2D
            ny  = ny_;  y0  = y0_;  y1 = y1_;
            Y   = new real[ny];
            dy = (y1-y0)/ny;       // cell-center
            for (int i=0;i<ny;  i++)	Y[i]  =  y0 + (i+0.5)*dy;
            #ifndef KO_ORD_3
            gy  = 3;
            #else
            gy  = 2;
            #endif
            // 2D v grid & cubature
            nv = gen_v2d_simple(nv_, vw, vy, vz);
            real dv = 2.0 / (nv_);
            long size = (ny+2*gy)*(nz+2*gz)*nv;
#else
            // dummy y-coor
            dy = dz;
            ny = 1;  y1  = dy;  y0 = 0;
            Y    = new real[ny];
            Y[0] = 0.5*dy;

            //nv = gen_v1d_vertex_center(nv_, vw, vz);
            //real dv = 2.0 / (nv_-1);
            nv = gen_v1d_cell_center(nv_, vw, vz);
            vy = new real[nv];
            real dv = 2.0 / (nv_);
            long size = (nz+2*gz)*nv;
#endif
            CFL = CFL_;
            dt = dz*CFL;


	    //====== Initial message...
#ifdef _OPENACC
            int ngpus = 0;	
            acc_device_t dev_type = acc_get_device_type();
    	    ngpus = acc_get_num_devices( dev_type ); 
            printf("\n\nOpenACC Enabled with GPU: %d.\n", ngpus) ;
            if (ngpus>1) printf("**Note: MultiGPU may not be effective currently.\n") ;
#endif

#ifdef COSENU2D
            printf("\nNuOsc2D with max OpenMP core: %d\n\n", omp_get_max_threads());
            printf("   Domain:  v: nv = %5d points within units disk, dv = %g\n", nv, dv);
            printf("            y:( %12f %12f )\t   ny  = %5d    buffer zone =%2d  dy = %g\n", y0,y1, ny, gy, dy);
#else
            printf("\nNuOsc1D with max OpenMP core: %d\n\n", omp_get_max_threads());
            printf("   Domain:  v:( -1 1 )\t   nv = %5d, dv = %g\n", nv, dv);
#endif
            printf("            z:( %12f %12f )\t   nz  = %5d    buffer zone =%2d  dz = %g\n", z0,z1, nz, gz, dz);
            printf("   Size per field var = %.1f MB\n", size*8/1024./1024.);
            printf("   dt = %g     CFL = %g\n", dt, CFL);

#ifndef KO_ORD_3
            printf("   5th order KO eps = %g\n", ko);
#else
            printf("   3th order KO eps = %g\n", ko);
#endif

#ifdef BC_PERI
            printf("   Use Periodic boundary\n");
#else
            printf("   Use open boundary -- Not impletemented yet.\n");
            assert(0);
#endif

#ifdef ADVEC_OFF
            printf("   No advection term.\n");
#else
  #if defined(ADVEC_LOPSIDED)
            printf("   Use lopsided FD for advaction\n");
  #elif defined(ADVEC_UPWIND)
            printf("   Use upwinded for advaction. (EXP. Always blowup!!\n");
  #else
            printf("   Use center-FD for advaction\n");
  #endif
#endif
            printf("\n");

            // supporting field variables for initial profile and analysis...
            G0     = new real[size];
            G0b    = new real[size];
            P1    = new real[size];
            P2    = new real[size];
            P3    = new real[size];
            P1b   = new real[size];
            P2b   = new real[size];
            P3b   = new real[size];
            dP  = new real[size];
            dN  = new real[size];
            dPb = new real[size];
            dNb = new real[size];

            // field variables...
            v_stat = new FieldVar(size);
            v_rhs  = new FieldVar(size);
            v_pre  = new FieldVar(size);
            v_cor  = new FieldVar(size);
#pragma acc enter data create(v_stat[0:1], v_rhs[0:1], v_pre[0:1], v_cor[0:1]) attach(v_stat, v_rhs, v_pre, v_cor)

            anafile.open("analysis.dat", std::ofstream::out | std::ofstream::trunc);
            if(!anafile) cout << "*** Open fails: " << "./analysis.dat" << endl;
            anafile << "### [ phy_time,   1:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6-8:aM0 ]" << endl;
        }

        ~NuOsc() {
            delete[] Z; delete[] vz; delete[] vw;
            delete[] Y; delete[] vy;
            delete[] G0;
            delete[] G0b;
            delete[] P1;  delete[] P2;  delete[] P3;  delete[] dP;  delete[] dN;
            delete[] P1b; delete[] P2b; delete[] P3b; delete[] dPb; delete[] dNb;
#pragma acc exit data delete(v_stat, v_rhs, v_pre, v_cor)
            delete v_stat, v_rhs, v_pre, v_cor;

            anafile.close();

        }

        int get_nv() { return nv;   }
        void set_mu(real mu_) {
            mu = mu_;
            printf("   Setting mu = %f\n", mu);
        }
        void set_renorm(bool renorm_) {
            renorm = renorm_;
            printf("   Setting renorm = %d\n", renorm);
        }


        void fillInitValue(int ipt, real alpha, real lnue, real lnueb, real eps0, real sigma);
        void updatePeriodicBoundary (FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);
        void step_rk4();
        void calRHS(FieldVar* out, const FieldVar * in);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
        void eval_extrinsics(const FieldVar* v0);
        void analysis();
        void renormalize(const FieldVar* v0);

	// Output:
	// TODO: should be able to use a single interface to generate arbitray slice over difference axis...
        void addSkimShot(std::list<real*> var, char *fntpl, int dumpstep, int sy, int sz, vector<int> vidx);
        void takeSkimShot(const int t=0);
        void takeSkimShotToConsole(const int t=0);
        void checkpoint(const int t = 0);
       
        // deprecated
        void snapshot(const int t = 0);
        FieldStat _analysis_v(const real var[]);
        FieldStat _analysis_c(const real vr[], const real vi[]);
	void output_detail(const char* fn);
        void __output_detail(const char* fn);

};
