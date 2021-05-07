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
    }
    ~Vars() {
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




class NuOsc {
    public:
        real phy_time, dt;

        // all coordinates
        real  *vz, *Z;
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

        inline unsigned int idx(const int i, const int j) { return i*(nz+2*gz) + (j+gz); }    // i:vz   j:z (last index)
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

            dz = (z1-z0)/nz;       // cell-center
            dt = dz*CFL;
            for (int i=0;i<nz;  i++)	Z[i]  =  z0 + (i+0.5)*dz;
#ifdef CELL_CENTER_V
            dv = (vz1-vz0)/nvz;      // for v as cell-center
            for (int i=0;i<nvz; i++)	vz[i] = vz0 + (i+0.5)*dv;   // cell-center
#else
            dv = (vz1-vz0)/(nvz-1);      // for v as vertex-center
            for (int i=0;i<nvz; i++)	vz[i] = vz0 + i*dv;   // vertex-center
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
            delete v_stat, v_rhs, v_pre, v_cor;

            anafile.close();

            ee_vl.close();    ee_vh.close();    ee_vm.close();
            p1_vm.close();  p2_vm.close();    p3_vm.close();
            p1_v.close();   p2_v.close();      p3_v.close();
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
        void fillInitValue_squared_wave();

        void updatePeriodicBoundary(FieldVar * in);
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
        void renormalize(FieldVar* v0);

        void write_fz();
        void write_bin(const int t);
        void output_detail(const char* fn);

};

void NuOsc::fillInitValue(real f0, real alpha, real lnue, real lnueb, int ipt, real eps0, real lzpt) {

    printf("   Init data: eps = %g  alpha = %f\n", eps0, alpha);

#pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {
            G0 [idx(i,j)] =         g(vz[i], 1.0, lnue )/2.0;
            G0b[idx(i,j)] = alpha * g(vz[i], 1.0, lnueb)/2.0;
        }
    }

#pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {
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

void NuOsc::fillInitValue_squared_wave() {
#pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=nz*0.4;j<nz*0.6; j++) {
            v_stat->ee    [idx(i,j)] = 1.0;
            v_stat->xx    [idx(i,j)] = 1.0;
            v_stat->ex_re [idx(i,j)] = 1.0;
            v_stat->ex_im [idx(i,j)] = 1.0;
            v_stat->bee   [idx(i,j)] = 1.0;
            v_stat->bxx   [idx(i,j)] = 1.0;
            v_stat->bex_re[idx(i,j)] = 1.0;
            v_stat->bex_im[idx(i,j)] = 1.0;
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

/*
void NuOsc::write_pn_bin(const int t, const int vreduction = 1, const int zreduction = 1) {
    char filename[32];
    sprintf(filename,"pn_%04d.bin", t);

    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }

    FieldVar *v = v_stat; 

    int reduced_nz = nz/zreduction;
    real P1tmp = new real[reduced_nz*5];  // only keep v=-1,-0.5,0,0.5,1


    outfile.write((char *) &phy_time,  sizeof(real));
    outfile.write((char *) &vz[0],     sizeof(real));
    outfile.write((char *) &vz[nvz-1], sizeof(real));
    outfile.write((char *) &Z[0],      sizeof(real));
    outfile.write((char *) &Z[nz-1],   sizeof(real));
    outfile.write((char *) &nz,  sizeof(int));
    outfile.write((char *) &nvz, sizeof(int));
    outfile.write((char *) P1,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) P2,  (nz+2*gz)*nvz*sizeof(real));
    outfile.write((char *) P3,  (nz+2*gz)*nvz*sizeof(real));
    outfile.close();
    printf("		Write %d x %d into %s\n", nvz, nz+2*gz, filename);
}
*/

void NuOsc::updatePeriodicBoundary(FieldVar * in) {

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<gz; j++) {
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
}

void NuOsc::updateInjetOpenBoundary(FieldVar * in) {
    // Cell-center for z:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]
    // Open boundary at two ends
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
	for (int j=0;j<gz; j++) {
        //lower open side, estimated simply by extrapolation
        in->ee    [idx(i,-j-1)] = in->ee    [idx(i,-j)]*2 - in->ee    [idx(i,-j+1)];
        in->xx    [idx(i,-j-1)] = in->xx    [idx(i,-j)]*2 - in->xx    [idx(i,-j+1)];
        in->ex_re [idx(i,-j-1)] = in->ex_re [idx(i,-j)]*2 - in->ex_re [idx(i,-j+1)];
        in->ex_im [idx(i,-j-1)] = in->ex_im [idx(i,-j)]*2 - in->ex_im [idx(i,-j+1)];
        in->bee   [idx(i,-j-1)] = in->bee   [idx(i,-j)]*2 - in->bee   [idx(i,-j+1)];
        in->bxx   [idx(i,-j-1)] = in->bxx   [idx(i,-j)]*2 - in->bxx   [idx(i,-j+1)];
        in->bex_re[idx(i,-j-1)] = in->bex_re[idx(i,-j)]*2 - in->bex_re[idx(i,-j+1)];
        in->bex_im[idx(i,-j-1)] = in->bex_im[idx(i,-j)]*2 - in->bex_im[idx(i,-j+1)];

        //upper open side, estimated simply by extrapolation
        in->ee    [idx(i,nz+j)] = in->ee    [idx(i,nz+j-1)]*2 - in->ee    [idx(i,nz+j-2)];
        in->xx    [idx(i,nz+j)] = in->xx    [idx(i,nz+j-1)]*2 - in->xx    [idx(i,nz+j-2)];
        in->ex_re [idx(i,nz+j)] = in->ex_re [idx(i,nz+j-1)]*2 - in->ex_re [idx(i,nz+j-2)];
        in->ex_im [idx(i,nz+j)] = in->ex_im [idx(i,nz+j-1)]*2 - in->ex_im [idx(i,nz+j-2)];
        in->bee   [idx(i,nz+j)] = in->bee   [idx(i,nz+j-1)]*2 - in->bee   [idx(i,nz+j-2)];
        in->bxx   [idx(i,nz+j)] = in->bxx   [idx(i,nz+j-1)]*2 - in->bxx   [idx(i,nz+j-2)];
        in->bex_re[idx(i,nz+j)] = in->bex_re[idx(i,nz+j-1)]*2 - in->bex_re[idx(i,nz+j-2)];
        in->bex_im[idx(i,nz+j)] = in->bex_im[idx(i,nz+j-1)]*2 - in->bex_im[idx(i,nz+j-2)];
	}
}

void NuOsc::calRHS(FieldVar * out, const FieldVar * in) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {

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
            #define ADV_FD(x)  (  (x[-2]-x[2]) - 8.0*(x[-1]-x[1]) )
#elif defined(ADVEC_UPWIND)
            int sv = sgn(vz[i]);
            real factor = -sv*vz[i]/(12*dz);
            #define ADV_FD(x)  ( -4*x[-3*sv] + 18*x[-2*sv] - 36*x[-sv] + 22*x[0] )
#else
            // advection term: (4-th order lopsided finite differencing)
            int sv = sgn(vz[i]);
            real factor = sv*vz[i]/(12*dz);
            #define ADV_FD(x)  ( x[-3*sv] - 6*x[-2*sv] + 18*x[-sv] - 10*x[0] - 3*x[sv] )
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
#ifdef CELL_CENTER_V
            for (int k=1;k<nvz; k++) {   // vz' integral
                real eep    = (in->ee    [idx(k,j)]);
                real xxp    = (in->xx    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bxxp   = (in->bxx   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=  2*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Ixx   += -2*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );  // = -Iee
                Iexr  +=    mu* (1-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=    mu* (1-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  2*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibxx  += -2*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) ); // = -Ibee
                Ibexr +=    mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=    mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
            }
#else
            for (int k=1;k<nvz-1; k++) {   // vz' integral
                real eep    = (in->ee    [idx(k,j)]);
                real xxp    = (in->xx    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bxxp   = (in->bxx   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=  2*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Ixx   += -2*mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );  // = -Iee
                Iexr  +=    mu* (1-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=    mu* (1-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  2*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibxx  += -2*mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) ); // = -Ibee
                Ibexr +=    mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=    mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
            }
            {   int k=0;  // Deal with end point for integral in vertex-center grid
                real eep    = (in->ee    [idx(k,j)]);
                real xxp    = (in->xx    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bxxp   = (in->bxx   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=  mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Ixx   += -mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );  // = -Iee
                Iexr  +=  0.5*mu* (1-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=  0.5* mu* (1-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibxx  += -mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) ); // = -Ibee
                Ibexr +=  0.5*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=  0.5*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
            }
            {   int k=nvz-1;  // Deal with end point for integral in vertex-center grid
                real eep    = (in->ee    [idx(k,j)]);
                real xxp    = (in->xx    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bxxp   = (in->bxx   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                Iee   +=  mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Ixx   += -mu* (1-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );  // = -Iee
                Iexr  +=  0.5*mu* (1-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=  0.5*mu* (1-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibxx  += -mu* (1-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) ); // = -Ibee
                Ibexr +=  0.5*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=  0.5*mu* (1-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
            }
#endif
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
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
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
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3) {
#pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
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
}

void NuOsc::eval_conserved(const FieldVar* v) {

    #pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        for (int j=0;j<nz; j++) {

            int ij=idx(i,j);
            P1   [ij] = v->ex_re[ij]/G0[ij];
	    P2   [ij] = v->ex_im[ij]/G0[ij];
	    P3   [ij] = (v->ee[ij] - v->xx[ij])/2.0/G0[ij];
	    P1b  [ij] = v->bex_re[ij]/G0b[ij];
            P2b  [ij] = v->bex_im[ij]/G0b[ij];
	    P3b  [ij] = (v->bee[ij] - v->bxx[ij])/2.0/G0b[ij];
	    relN [ij] = ((v->ee [ij] + v->xx [ij]) / (2.0*G0 [ij])) - 1.0;
	    relNb[ij] = ((v->bee[ij] + v->bxx[ij]) / (2.0*G0b[ij])) - 1.0;
	    relP [ij] = sqrt(P1 [ij]*P1 [ij]+P2 [ij]*P2 [ij]+P3 [ij]*P3 [ij]) - 1.0;
	    relPb[ij] = sqrt(P1b[ij]*P1b[ij]+P2b[ij]*P2b[ij]+P3b[ij]*P3b[ij]) - 1.0;
        }
    }
}

void NuOsc::renormalize(FieldVar* v0) {
    #pragma omp parallel for
    for(int i=0; i<nvz; i++)
    for(int j=0; j<nz; j++) {
	int ij=idx(i,j);
	real iG = 1.0 / G0[ij];
	real iGb = 1.0 / G0b[ij];
	real P1  = v0->ex_re[ij] * iG;
        real P2  = v0->ex_im[ij] * iG;
        real P3  = (v0->ee[ij] - v0->xx[ij])*iG*0.5;
        real P1b = v0->bex_re[ij] * iGb;
        real P2b = v0->bex_im[ij] * iGb;
        real P3b = (v0->bee[ij] - v0->bxx[ij])*0.5 * iGb;
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

    if(renorm) renormalize(v_stat);

    phy_time += dt;
}

FieldStat NuOsc::_analysis_v(const real var[]) {

    FieldStat res;
    
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
    res.min = vmin;
    res.max = vmax;
    res.sum = sum;
    res.avg = sum/(nz*nvz);
    res.std = sqrt( sum2/(nz*nvz) - res.avg*res.avg  );

    return res;
}

FieldStat NuOsc::_analysis_c(const real vr[], const real vi[]) {

    FieldStat res;
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
    res.min = vmin;
    res.max = vmax;
    res.sum = sum;
    res.avg = sum/(nz*nvz);
    res.std = sqrt( sum2/(nz*nvz) - res.avg*res.avg  );

    return res;
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

void NuOsc::output_detail(const char* filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }
    outfile << "#phy_time=" << phy_time << endl;
    int iskip=nz/100;
    for(int i=0;i<nvz;i++){
    for(int j=0;j<nz;j=j+iskip){
    int ij=idx(i,j);
    outfile << vz[i] << " " << Z[j] << " " << P1 << " " << P2 << " " << P3 << endl;
     //        outfile << vz[i] << " " << Z[j] << " " << v_stat->ee[ij] << " " << v_stat->ex_re[ij] << " " << v_stat->ex_im[ij] << endl;
    }
    outfile << endl;
    }
}

void NuOsc::analysis() {

    eval_conserved(v_stat);

    real maxrelP = 0.0;
    real maxrelN = 0.0;
    real avgP  = 0.0;
    real avgPb = 0.0;
    real norP  = 0.0;
    real norPb = 0.0;
    real aM11 = 0.0, aM12 = 0.0, aM13 = 0.0;
    real aM01 = 0.0, aM02 = 0.0, aM03 = 0.0;
    real nor=0.0;
    //int maxi,maxj;

    #pragma omp parallel for reduction(+:avgP,avgPb,aM01,aM02,aM03,aM11,aM12,aM13,norP,norPb,nor) reduction(max:maxrelP,maxrelN)
    for(int i=0;i<nvz;i++)
    for(int j=0;j<nz;j++)  {
	int ij = idx(i,j);
        //if (relP>maxrelP || relPb>maxrelP) {maxi=i;maxj=j;}
        maxrelP = std::max( std::max(maxrelP,relP[ij]), relPb[ij]);
        maxrelN = std::max( std::max(maxrelN,relN[ij]), relNb[ij]);
        
        avgP  += v_stat->ee [ij];
        avgPb += v_stat->bee[ij];
        aM01  += v_stat->ex_re[ij] - v_stat->bex_re[ij];                                    // P1[ij]*G0[ij] - P1b[ij]*G0b[ij];
        aM02  += v_stat->ex_im[ij] - v_stat->bex_im[ij];                                    // P2[ij]*G0[ij] - P2b[ij]*G0b[ij];
        aM03  += 0.5*(v_stat->ee[ij] - v_stat->ee[ij] - v_stat->bee[ij] + v_stat->bxx[ij]); // P3[ij]*G0[ij] - P3b[ij]*G0b[ij];
        aM11  += vz[i]*(v_stat->ex_re[ij] - v_stat->bex_re[ij]);
        aM12  += vz[i]*(v_stat->ex_im[ij] - v_stat->bex_im[ij]);
        aM13  += vz[i]*0.5*(v_stat->ee[ij] - v_stat->xx[ij] - v_stat->bee[ij] + v_stat->bxx[ij]);
        norP  += 2.0*G0 [ij];
        norPb += 2.0*G0b[ij];
        nor   += 2.0*(G0[ij]-G0b[ij]);
    }
    avgP  /= norP;
    avgPb /= norPb;
    real aM0    = sqrt(aM01*aM01+aM02*aM02+aM03*aM03)/nor;
    real aM1    = sqrt(aM11*aM11+aM12*aM12+aM13*aM13)/nor;
    printf("T= %15f ", phy_time);
    printf("%6.5e %6.5e %6.5e %6.5e %6.5e %6.5e\n",avgP,avgPb,maxrelP,maxrelN,aM0, aM1);
    
    anafile << phy_time << " " << avgP << " " << avgPb << " " << maxrelP << " " << maxrelN << " " << aM0 << " " << aM1 << endl;
}

void NuOsc::write_fz() {
    FieldVar *v = v_stat;
#define WRITE_Z_AT(HANDLE, VAR, V_IDX) \
        HANDLE << phy_time << " "; \
        for (int i=0;i<nz; i++) HANDLE << std::setprecision(8) << VAR[idx(V_IDX, i)] << " "; \
        HANDLE << endl;
    // f(z) at the highest v-mode
    //WRITE_Z_AT(ee_vh,  v->ee,    nvz-1)

{
    // Pn at v = -0.5
    int at_vz = int(0.25*(nvz-1));
    WRITE_Z_AT(p1_vm,   P1,  at_vz);
    WRITE_Z_AT(p2_vm,   P2,  at_vz);
    WRITE_Z_AT(p3_vm,   P3,  at_vz);
}
{    // Pn at v = 1
    int at_vz = nvz-1;
    WRITE_Z_AT(p1_v,   P1,  at_vz);
    WRITE_Z_AT(p2_v,   P2,  at_vz);
    WRITE_Z_AT(p3_v,   P3,  at_vz);
}
#undef WRITE_Z_AT

}

int main(int argc, char *argv[]) {

#ifdef PAPI
    if ( PAPI_hl_region_begin("computation") != PAPI_OK )
       cout << "PAPI error!" << endl;
#endif

    real dz  = 0.2;
    real z0  = -600;     real z1  =  -z0;
    real vz0 = -1;       real vz1 =  -vz0;    int nvz = 16 + 1;
    real cfl = 0.4;      real ko = 0.0;

    real mu  = 0.0;
    bool renorm = false;

    // === initial value
    real alpha = 0.9;     //0.92 for G4b  // nuebar/nue_asymmetric_parameter
    real lnue  = 0.6;     // width_nue
    real lnueb = 0.53;    // width_nuebar
    real ipt   = 0;       // 0_for_central_z_perturbation;1_for_random;2_for_perodic
    real eps0  = 0.1;     // 1e-7 for G4b    // eps0
    real lzpt  = 50.0;    // width_pert_for_0

    int ANAL_EVERY = 10.0   / (cfl*dz) + 1;
    int END_STEP   = 900.0 / (cfl*dz) + 1;
    int DUMP_EVERY = 99999999;

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
	    #ifdef CELL_CENTER_V
            assert(nvz%2==0);
            #else
            assert(nvz%2==1);
            #endif
        } else if (strcmp(argv[t], "--ko") == 0 )  {
            ko    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--mu") == 0 )  {
            mu    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--renorm") == 0 )  {
            renorm = bool(atoi(argv[t+1]));    t+=1;
        
        // for monitoring
        } else if (strcmp(argv[t], "--ANA_EVERY") == 0 )  {
            ANAL_EVERY = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--DUMP_EVERY") == 0 )  {
            DUMP_EVERY = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ENDSTEP") == 0 )  {
            END_STEP = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ANA_EVERY_T") == 0 )  {
            ANAL_EVERY = int( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
        } else if (strcmp(argv[t], "--DUMP_EVERY_T") == 0 )  {
            DUMP_EVERY = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
        } else if (strcmp(argv[t], "--ENDSTEP_T") == 0 )  {
            END_STEP = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
        // for intial data
        } else if (strcmp(argv[t], "--lnue") == 0 )  {
            lnue  = atof(argv[t+1]);
            lnueb = atof(argv[t+2]);    t+=2;
        } else if (strcmp(argv[t], "--eps0") == 0 )  {
            eps0 = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--alpha") == 0 )  {
            alpha = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ipt") == 0 )  {
            ipt = atoi(argv[t+1]);    t+=1;
        } else {
            printf("Unreconganized parameters %s!\n", argv[t]);
            exit(0);
        }
    }
    int nz  = int((z1-z0)/dz);

    // === Initialize simuation
    NuOsc state(nvz, nz, vz0, vz1, z0, z1, cfl, ko);
    state.set_mu(mu);
    state.set_renorm(renorm);

    state.fillInitValue(1.0, alpha, lnue, lnueb, ipt, eps0, lzpt);
    
    // === analysis for t=0
    state.analysis();
    state.write_fz();
    //state.write_bin(0);

    for (int t=1; t<=END_STEP; t++) {
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            state.analysis();
        }
        if ( t%DUMP_EVERY==0) {
            state.write_fz();
            //state.write_bin(t);
        }
    }

#ifdef PAPI
    if ( PAPI_hl_region_end("computation") != PAPI_OK )
       cout << "PAPI error!" << endl;
#endif

    printf("Completed.\n");
    return 0;
}








