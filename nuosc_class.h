#ifndef _NUOSC_CLASS_H_
#define _NUOSC_CLASS_H_

//==== Major configuration macros to use  =========
#define BC_PERI
#define VACUUM_OFF
#define VERTEX_CENTER_V
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

#define COLLAPSE_LOOP 3

#define PARFORALL(i,p,v) \
        _Pragma("omp parallel for collapse(3)") \
        _Pragma("acc parallel loop collapse(3)") \
        for (int i=0;i<nz; ++i) \
        for (int p=0;p<np; ++p) \
        for (int v=0;v<nv; ++v)

#define FORALL(i,p,v) \
                for (int i=0;i<nz; ++i) \
                for (int p=0;p<np; ++p) \
                for (int v=0;v<nv; ++v)
                                
                                

using std::cout;
using std::endl;
using std::cin;
using std::string;

typedef double real;

#include "jacobi_poly.h"

// dG stuff
// See implementation in Appendix A of the dG text book.

typedef struct Vars {
	std::vector<real> ee, xx, ex_re, ex_im;
	std::vector<real> bee, bxx, bex_re, bex_im;

	Vars(int size) {
		ee.resize(size);
		xx.resize(size);
		ex_re.resize(size);
		ex_im.resize(size);
		bee.resize(size);
		bxx.resize(size);
		bex_re.resize(size);
		bex_im.resize(size);
#pragma acc enter data create(this,ee[0:size],xx[0:size],ex_re[0:size],ex_im[0:size],bee[0:size],bxx[0:size],bex_re[0:size],bex_im[0:size])
	}
	~Vars() {
#pragma acc exit data delete(ee, xx, ex_re, ex_im, bee, bxx, bex_re, bex_im, this)
	}
} FieldVar;

typedef struct stat {
	real min;
	real max;
	real sum;
	real avg;
	real std;
} FieldStat;


typedef struct SnapShot_struct {
	std::list<real*> var_list;
	string fntpl;
	int every;
	std::vector<int> y_slices;   // coordinate for the reduced dimension
	std::vector<int> v_slices;

	// init with specified v-coordinate
	SnapShot_struct(std::list<real*> var_list_, string fntpl_, uint every_,  std::vector<int> v_slices_) {
		var_list = var_list_;
		fntpl = fntpl_;
		every = every_;
		v_slices = v_slices_;
	}
	// init with specified y and v-coordinate
	SnapShot_struct(std::list<real*> var_list_, string fntpl_, uint every_, std::vector<int> y_slices_, std::vector<int> v_slices_) {
		var_list = var_list_;
		fntpl = fntpl_;
		every = every_;
		y_slices = y_slices_;
		v_slices = v_slices_;
	}

} SnapShot;


inline void swap(FieldVar **a, FieldVar **b) { FieldVar *tmp = *a; *a = *b; *b = tmp; }
inline real random_amp(real a) { return a * rand() / RAND_MAX; }
template <typename T> int sgn(T val) {    return (T(0) < val) - (val < T(0));   }

int gen_v2d_simple_vertex_center(const int nv_, real *& vw, real *& vy, real *& vz);
int gen_v2d_simple_cell_center(const int nv_, real *& vw, real *& vy, real *& vz);
int gen_v1d_vertex_center(const int nv_, real *& vw, real *& vz);
int gen_v1d_cell_center(const int nv_, real *& vw, real *& vz);

std::vector<int> gen_skimmed_vslice_index(uint nv_target, uint nv_in, const real * vy, const real * vz);
std::vector<int> gen_skimmed_vslice_index(uint nv_target, uint nv_in);


class NuOsc {
public:
	real phy_time, dt;
	
	// dG stuff
	uint ord, np;  // dG order, will allocate np=ord+1 quadruture pts in each element
	std::vector<real> r;  // Gauss-Lobatto quadrature points
	std::vector<real> w;  // GL weights
	std::vector<std::vector<real>> V;     // Vondermon martix
	std::vector<std::vector<real>> Vinv;
	std::vector<std::vector<real>> Dr;   // Differential matrix
	std::vector<std::vector<real>> VVT;

        // v-coor
	real *vw;   // integral quadrature
	int nv;      // # of v cubature points.

	// z-coordinate
	real *vz, *Z;
	real z0, z1, dz;
	int nz;  // num of z elements.

	FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;  // field variables

	real *P1,  *P2,  *P3,  *dN,  *dP;
	real *P1b, *P2b, *P3b, *dNb, *dPb;
	real *G0,*G0b;
        real n_nue0,n_nueb0;   // initial number density for nue
        
	real CFL;

	const real theta = 1e-6;
	const real ct = cos(2*theta);
	const real st = sin(2*theta);
	const int  pmo = 0; // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
	real mu = 1.0;      // can be set by set_mu()
	bool renorm = false;  // can be set by set_renorm()
	
	real flux_alpha = 0.0;  // Upwind
	//real flux_alpha = 1.0;  // central-flux

	std::ofstream anafile;
	std::list<SnapShot> snapshots;
	
	inline unsigned long idx(const int i, const int p) const              { return   i*np + p; }
	inline unsigned long idx(const int i, const int p, const int v) const { return   ( i*np + p)*nv + v; }

	NuOsc(const int  ord_, const int  nv_, 
	const int   nz_,
	const real  z0_, const real  z1_,
	const real CFL_) : phy_time(0.)  {

		//
		// dG stuff
		//
		ord   = ord_;   np = ord + 1;
		r.resize(np);
		w.resize(np);
		JacobiGL(ord,0,0,r,w);
		V    = std::vector<std::vector<real>> { static_cast<size_t>(np), std::vector<double>(static_cast<size_t>(np), 0.0) };
		Vinv = std::vector<std::vector<real>> { static_cast<size_t>(np), std::vector<double>(static_cast<size_t>(np), 0.0) };
		Vandermonde1D(ord, r, V, np);
		InvMatrix(np, V, Vinv);
		Dr = std::vector<std::vector<real>> { static_cast<size_t>(np), std::vector<double>(static_cast<size_t>(np), 0.0) };
		Dmatrix1D(ord, r, V, Dr, np);

#if 0
    cout << "=== Matrix : " << V.size() << " x " << V[0].size() << endl;
    for (auto X:  V) {
       for (auto v : X) cout << v << " ";
       cout << endl;
    }
#endif
		VVT = std::vector<std::vector<real>> { static_cast<size_t>(np), std::vector<double>(2, 0.0) };
		for (int p=0;p<np; ++p)
		for (int k=0;k<np; ++k) {
		    VVT[p][0]  +=  V[p][k]*V[0   ][k];
		    VVT[p][1]  +=  V[p][k]*V[np-1][k];
		}

#if 0
    cout << "====== VVT : " << VVT.size() << " x " << VVT[0].size() << endl;
    for (auto X:  VVT) {
       for (auto v : X) cout << v << " ";
       cout << endl;
    }
#endif

		// space coordinates
		nz  = nz_;  z0  = z0_;  z1 = z1_;
		Z      = new real[nz*np];
		dz = (z1-z0)/nz;       // cell-center
		for (int i=0;i<nz; ++i)
		for (int p=0;p<np; ++p)
		    Z[i*np+p]  =  z0 + (i + 0.5*(r[p]+1)) * dz;

#ifdef VERTEX_CENTER_V
		nv = gen_v1d_vertex_center(nv_, vw, vz);
		real dv = 2.0 / (nv_-1);
#else
		nv = gen_v1d_cell_center(nv_, vw, vz);
		real dv = 2.0 / (nv_);
#endif
		long size = nz*np*nv;

                double mindz = 1e10;
                {   // find min dz and set dt
		for (int p=0;p<np-1; ++p) {
		    double dd =  Z[p+1] - Z[p];
		    if (dd < mindz)  mindz = dd;
		}    
		CFL = CFL_;
		dt = CFL*mindz;
                }

		//====== Initial message...
#ifdef _OPENACC
		int ngpus = 0;	
		acc_device_t dev_type = acc_get_device_type();
		ngpus = acc_get_num_devices( dev_type ); 
		printf("\n\nOpenACC Enabled with GPU: %d.\n", ngpus) ;
		if (ngpus>1) printf("**Note: MultiGPU may not be effective currently.\n") ;
#endif

		printf("\nNuOsc1D with max OpenMP core: %d\n\n", omp_get_max_threads());
		printf("   Domain:  v:( -1 1 )\t   nv = %5d, dv = %g\n", nv, dv);
		printf("            z:( %12f %12f )\t   nz  = %5d    np = %2d  dz = %g  mindz = %f\n", z0,z1, nz, np, dz, mindz);
		printf("   Size per field var = %.2f MB\n", size*8/1024./1024.);
		printf("   dt = %g     CFL = %g\n", dt, CFL);

#ifdef BC_PERI
		printf("   Use Periodic boundary\n");
#else
		printf("   Use open boundary -- Not impletemented yet.\n");
		assert(0);
#endif

#ifdef ADVEC_OFF
		printf("   No advection term.\n");
#else
		printf("   Use dG.\n");
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
		delete[] G0;
		delete[] G0b;
		delete[] P1;  delete[] P2;  delete[] P3;  delete[] dP;  delete[] dN;
		delete[] P1b; delete[] P2b; delete[] P3b; delete[] dPb; delete[] dNb;
#pragma acc exit data delete(v_stat, v_rhs, v_pre, v_cor)
		delete v_stat, v_rhs, v_pre, v_cor;

		anafile.close();

	}

	inline int get_nv() const { return nv;   }
	inline int get_np() const { return np;   }

	inline real get_dt() const { return dt;   }
	inline void set_dt(real dt_)  {
	    dt = dt_;
	    printf("   Setting dt = %f\n", dt);
	}
	
	inline void set_mu(real mu_) {
		mu = mu_;
		printf("   Setting mu = %f\n", mu);
	}
	inline void set_renorm(bool renorm_) {
		renorm = renorm_;
		printf("   Setting renorm = %d\n", renorm);
	}


        void fillInitGaussian(real eps0, real sigma);
        void fillInitSquare(real eps0, real sigma);
        void fillInitTriangle(real eps0, real sigma);
	void fillInitValue(int ipt, real alpha, real lnue, real lnueb, real eps0, real sigma);
	void updatePeriodicBoundary (FieldVar * in);
	void updateInjetOpenBoundary(FieldVar * in);
	void step_rk4();
	void calRHS(FieldVar* out, const FieldVar * in);
	void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
	void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
	void eval_extrinsics(const FieldVar* v0);
	void analysis();
	void renormalize(FieldVar* const v0);

	// 1D output:
	void addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int>  vidx);
	void takeSnapShot(const int t=0) const;

	// Output:
	// TODO: should be able to use a single interface to generate arbitray slice over difference axis...
	//void addSkimShot(std::list<real*> var, char *fntpl, int dumpstep, int sy, int sz, std::vector<int> vidx);
	//void takeSkimShot(const int t=0);
	//void takeSkimShotToConsole(const int t=0);
	//void checkpoint(const int t = 0);
	
        // deprecated
        void snapshot(const int t = 0);
        FieldStat _analysis_v(const real var[]);
        FieldStat _analysis_c(const real vr[], const real vi[]);
	void output_detail(const char* fn);
	void __output_detail(const char* fn);
	void checkpoint(const int t);
};

#endif