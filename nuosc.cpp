#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>

#define ADVEC_OFF
//#define MUINT_OFF
#define BC_PERI

using std::cout;
using std::endl;
using std::cin;

typedef double real;

inline real random_amp(real a) { return a * rand() / RAND_MAX; }

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

class NuOsc {
    public:
        real phy_time, dt;

        // all field variables...
        real     *Z, *vz;
        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;

        int nz;  // Dim of z  (the last dimension, the one with derivatives. Cell-center grid used.)
        int nvz; // Dim of vz (Vertex-center grid used.)
        int gz;  // Width of z-buffer zone. 4 for 2nd-order of d/dz.
        real dz, dv;
        real vz0, vz1;
        real  z0,  z1;

        real mu = 100;
        const real theta = 0.01;
        const real ct = cos(2*theta);
        const real st = sin(2*theta);

	const float RND = 1.e-7;
        
        std::ofstream anafile;
        std::ofstream outfile;

        inline unsigned int idx(const int i, const int j) { return i*(nz+2*gz) + (j+gz); }
        inline unsigned int idz(const int j) { return j; }


        NuOsc(const int nz_, const int nvz_, const int gz_ = 2) : phy_time(0.)  {

	    //vz0 = -1.0;  vz1 =  1.0;
	    //z0 = -1.0;   z1 =  1.0;
	    vz0 = .0;  vz1 =  1.0;
	    z0 = .0;   z1 =  1.0;
            nz  = nz_;
            gz  = gz_;
            nvz = nvz_;
            int size = (nz+2*gz)*(nvz);
            Z      = new real[nz];
            vz     = new real[nvz];

            v_stat = new FieldVar(size);
            v_rhs  = new FieldVar(size);
            v_pre  = new FieldVar(size);
            v_cor  = new FieldVar(size);

            real CFL = 0.05;
            dz = (z1-z0)/nz;       // cell-center
	    dt = CFL*dz/vz1;
	    
	    dv = (vz1-vz0)/(nvz-1.0);      // we let v as vertex-center

	    for (int i=0;i<nz;  i++)	Z[i]  = z0 + (i+0.5)*dz;
	    for (int i=0;i<nvz; i++)	vz[i] = vz0 + i*dv;
	    
	    printf("Initializing simulation NuOsc :\n\n");
	    printf("   Domain: z:(0 1)[fixd] vz:(%f %f)\n", vz[0], vz[nvz-1]);
	    printf("   (Nvz, Nz) = (%d, %d) with z-buffer zone %d.\n", nz, nvz, gz);
	    printf("   dz = %g\n", dz);
	    printf("   dt = %g\n\n", dt);
	    printf("========================\n\n");

	    anafile.open("rate.dat", std::ofstream::out | std::ofstream::trunc);
	    if(!anafile) cout << "*** Open fails: " << "./rate.dat" << endl;
	    outfile.open("writez.dat", std::ofstream::out | std::ofstream::trunc);
	    if(!anafile) cout << "*** Open fails: " << "./writez.dat" << endl;
        }

        //NuOsc(const NuOsc &v) {  // Copy constructor to be checked.
        //    NuOsc(v.nz, v.nvz);
        //}

        ~NuOsc() {
            delete[] vz;
            delete v_stat, v_rhs, v_pre, v_cor;
            
            anafile.close();
            outfile.close();
        }

        void set_mu(real mu_) { mu = mu_; }
        
        void fillInitValue(real f0, real alpha, real beta);
        void updatePeriodicBufferZone(FieldVar * in);
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
        void write_bin(const int t);

    inline void _out_tmp(real res[], const real v[]) {
        res[0] = v[idx(0, 1)];
        res[1] = v[idx(50, 1)];
        res[2] = v[idx(100, 1)];
    }

};

void NuOsc::fillInitValue(real f0 = 1.0, real alpha=2.0, real beta=0.5) {

    // Init value
    #pragma omp parallel for
    for (int i=0;i<nvz; i++) {
          for (int j=0;j<nz; j++) {
            //v_stat->ee    [idx(i,j)] = 1.0*exp(-Z[j]*Z[j]*100);    // Initialize with Gaussian shape
            v_stat->ee    [idx(i,j)] = 0.99;    // Initialize with Gaussian shape
            v_stat->ex_re [idx(i,j)] = random_amp(RND);
            v_stat->ex_im [idx(i,j)] = random_amp(RND);
            v_stat->bee   [idx(i,j)] = random_amp(RND);
            v_stat->bex_re[idx(i,j)] = random_amp(RND);
            v_stat->bex_im[idx(i,j)] = random_amp(RND);
          }
    }

    int nvz_b = int(beta*nvz);
    #pragma omp parallel for
    for (int i=nvz_b;i<nvz; i++) {
          for (int j=0;j<nz; j++) {
            v_stat->bee   [idx(i,j)] = alpha;
            v_stat->bex_re[idx(i,j)] = 0.0;
            v_stat->bex_im[idx(i,j)] = 0.0;
          }
    }



    // Init boundary
    #ifdef BC_PERI
    updatePeriodicBufferZone(v_stat);
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

void NuOsc::updatePeriodicBufferZone(FieldVar * in) {
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

    /*  Constant injection at z~0.
    #pragma omp parallel for
    for (int i=0;i<nvz; i++) {
        in->ee    [idx(i,nz/2)]    = 0.8;
        in->ee    [idx(i,nz/2+1)]  = 0.8;
        in->bee   [idx(i,nz/2)]    = 0.4;
        in->bee   [idx(i,nz/2+1)]  = 0.4;
    }
    */

    // Open boundary at two ends
    #pragma omp parallel for
    for (int i=0;i<nvz; i++) {
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
            out->ee    [idx(i,j)] = -2*st*exi [0];
            out->ex_re [idx(i,j)] = -2*ct*exi [0];
            out->ex_im [idx(i,j)] =  2*ct*exr [0] + st*( 2*ee[0] -1 );
            out->bee   [idx(i,j)] = -2*st*bexi[0];
            out->bex_re[idx(i,j)] = -2*ct*bexi[0];
            out->bex_im[idx(i,j)] =  2*ct*bexr[0] + st*( 2*bee[0]-1 );

#ifndef ADVEC_OFF
            // 2) advection term:
            //   4-th order FD for 2nd-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor = -vz[i]/(12*dz);
            out->ee    [idx(i,j)] += factor * ((   ee[-2]-  ee[2]) - 8.0*(  ee[-1]-  ee[1]) );
            out->ex_re [idx(i,j)] += factor * ((  exr[-2]- exr[2]) - 8.0*( exr[-1]- exr[1]) );
            out->ex_im [idx(i,j)] += factor * ((  exi[-2]- exi[2]) - 8.0*( exi[-1]- exi[1]) );
            out->bee   [idx(i,j)] += factor * ((  bee[-2]- bee[2]) - 8.0*( bee[-1]- bee[1]) );
            out->bex_re[idx(i,j)] += factor * (( bexr[-2]-bexr[2]) - 8.0*(bexr[-1]-bexr[1]) );
            out->bex_im[idx(i,j)] += factor * (( bexi[-2]-bexi[2]) - 8.0*(bexi[-1]-bexi[1]) );
#endif

#ifndef MUINT_OFF
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
#endif
        }
}

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
    updatePeriodicBufferZone(v_stat);
    #else
    updateInjetOpenBoundary(v_stat);
    #endif
    calRHS(v_rhs, v_stat);
    vectorize(v_pre, v_stat, 0.5*dt, v_rhs);

    //Step-2
    #ifdef BC_PERI
    updatePeriodicBufferZone(v_pre);
    #else
    updateInjetOpenBoundary(v_pre);
    #endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, 0.5*dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-3
    #ifdef BC_PERI
    updatePeriodicBufferZone(v_pre);
    #else
    updateInjetOpenBoundary(v_pre);
    #endif
    calRHS(v_cor, v_pre);
    vectorize(v_rhs, v_rhs, 2.0, v_cor);
    vectorize(v_cor, v_stat, dt, v_cor);
    swap(&v_pre, &v_cor);

    //Step-4
    #ifdef BC_PERI
    updatePeriodicBufferZone(v_pre);
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
    real Iee    = 0.5*sqrt(vr[idx(loc,0)]*vr[idx(loc,0)]+vi[idx(loc,0)]*vi[idx(loc,0)]);
    Iee        += 0.5*sqrt(vr[idx(loc,nvz-1)]*vr[idx(loc,nvz-1)]+vi[idx(loc,nvz-1)]*vi[idx(loc,nvz-1)]);
    #pragma omp parallel for
    for (int k=1;k<nvz-1; k++) {   // vz' integral
	Iee   += sqrt(vr[idx(loc,k)]*vr[idx(loc,k)]+vi[idx(loc,k)]*vi[idx(loc,k)]);
    }

    res = dv*Iee;
}

void NuOsc::analysis() {

    //real statis1[4], statis2[4];
    //_analysis_c(statis1, v_stat-> ex_re, v_stat-> ex_im);
    //_analysis_v(statis2, v_stat->bee);

    //printf("Time: %.5f  min/max/std of |ex|:( %9.2g %9.2g %9.2g ) |bee|:( %9.2g %9.2g %9.2g)\n", phy_time,
    //			statis1[0], statis1[1], statis1[3],
    //			statis2[0], statis2[1], statis2[3]);
    //anafile << phy_time <<" "<< statis1[0]<< " " << statis1[1] << " " << statis1[3] << " "
    //                         << statis2[0]<< " " << statis2[1] << " " << statis2[3] << " "
    //                         << probe0 << endl;

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
    outfile << phy_time << " ";
    for (int i=0;i<nz; i++) outfile << v->ee[idx(nvz/2+10, i)] << " ";
    outfile << endl;
}

int main(int argc, char *argv[]) {

    int END_TIME   = 500;
    int DUMP_EVERY = 1000;
    int ANAL_EVERY = 1;
    
    int nz  = 8;
    int nvz = 256 + 1;
    
    real f0    = 1.0;
    real alpha = 2.0;
    real mu = 100;

    // TODO: Parse input argument
    for (int t = 1; argv[t] != 0; t++) {
	if (strcmp(argv[t], "--dim") == 0 )  {
	    nz  = atoi(argv[t+1]);
	    nvz = atoi(argv[t+2]);     t+=2;
	} else if (strcmp(argv[t], "--time") == 0 )  {
	    END_TIME   = atoi(argv[t+1]); t+=1;
	} else if (strcmp(argv[t], "--ana") == 0 )  {
	    DUMP_EVERY = atoi(argv[t+1]); t+=1;
	} else if (strcmp(argv[t], "--f0") == 0 )  {
	    f0 = atof(argv[t+1]); t+=1;
	} else if (strcmp(argv[t], "--alpha") == 0 )  {
	    alpha = atof(argv[t+1]); t+=1;
	} else if (strcmp(argv[t], "--mu") == 0 )  {
	    mu = atof(argv[t+1]); t+=1;
	} else {
	    printf("Unreconganized parameters %s!\n", argv[t]);
	    exit(0);
	}
    }

    // Initialize simuation
    NuOsc state(nz, nvz);
    state.set_mu(mu);
    
    // initial value
    state.fillInitValue(f0, alpha);
    state.write_bin(0);

    for (int t=1; t<=END_TIME; t++) {
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
    	    state.analysis();
    	    //state.write_z_at_vz();
    	}
        if ( t%DUMP_EVERY==0)  state.write_bin(t);
    }

    printf("Completed.\n");
    return 0;
}
