#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>

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
        real     *vz;
        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;

        int nz;  // Dim of z  (the last dimension, the one with derivatives. Cell-center grid used.)
        int nvz; // Dim of vz (Vertex-center grid used.)
        int gz;  // Width of z-buffer zone. 4 for 2nd-order of d/dz.
        real dz, dv;

        const real theta = 0.01;
        const real ct = cos(theta);
        const real st = sin(theta);
        const real mu = 100;

        std::ofstream anafile;


        inline unsigned int idx(const int i, const int j) { return i*(nz+2*gz) + (j+gz); }



        NuOsc(const int nz_, const int nvz_, const int gz_ = 2) : phy_time(0.)  {

            nz  = nz_;
            gz  = gz_;
            nvz = nvz_;
            int size = (nz+2*gz)*(nvz);
            vz     = new real[nvz];

            v_stat = new FieldVar(size);
            v_rhs  = new FieldVar(size);
            v_pre  = new FieldVar(size);
            v_cor  = new FieldVar(size);
            
            //const real CFL = 0.01;
            const real CFL = 0.1;
            dz = real(1.0)/nz;       // cell-center
	    dv = 1.0/(nvz-1.0);  // vertex-center
	    dt = CFL*dz;
	    
	    printf("Initializing simulation NuOsc :\n\n");
	    printf("   (Nvz, Nz) = (%d, %d) with z-buffer zone %d.\n", nz, nvz, gz);
	    printf("   dz = %g\n", dz);
	    printf("   dt = %g\n\n", dt);
	    printf("========================\n\n");

	    anafile.open("rate.dat", std::ofstream::out | std::ofstream::trunc);
	    if(!anafile) cout << "*** Open fails: " << "./rate.dat" << endl;
        }

        //NuOsc(const NuOsc &v) {  // To be checked.
        //    NuOsc(v.nz, v.nvz);
        //}

        ~NuOsc() {
            delete[] vz;
            delete v_stat, v_rhs, v_pre, v_cor;
            
            anafile.close();
        }

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
        void dumpv(const real v[]);
        void write_bin(const char* fn);
};

void NuOsc::fillInitValue(real f0 = 1.0, real alpha=2.0, real beta=0.5) {
    // Init value
    for (int i=0;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            vz[i] = real(i)/nvz;   // only vz is vertex-center

            v_stat->ee    [idx(i,j)] = f0;
            v_stat->ex_re [idx(i,j)] = random_amp(0.001);
            v_stat->ex_im [idx(i,j)] = random_amp(0.001);
            v_stat->bee   [idx(i,j)] = f0;
            v_stat->bex_re[idx(i,j)] = random_amp(0.001);
            v_stat->bex_im[idx(i,j)] = random_amp(0.001);
        }

    int nvz_b = int(beta*nvz);
    for (int i=nvz_b;i<nvz; i++)
        for (int j=0;j<nz; j++) {
            v_stat->ee [idx(i,j)]  = alpha;
            v_stat->bee[idx(i,j)]  = alpha;
        }

    // Init boundary
    #ifdef BC_PERI
    updatePeriodicBufferZone(v_stat);
    #else
    updateInjetOpenBoundary(v_stat);
    #endif
}

void NuOsc::write_bin(const char* filename) {
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }

    FieldVar *v = v_stat; 

    outfile.write((char *) &phy_time, sizeof(real));
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
    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

    #pragma omp parallel for
    for (int i=0;i<nvz; i++)
        for (int j=0;j<gz; j++) {
            //lower side
            in->ee    [idx(i,-j-1)] = in->ee    [idx(i,0)];
            in->ex_re [idx(i,-j-1)] = in->ex_re [idx(i,0)];
            in->ex_im [idx(i,-j-1)] = in->ex_im [idx(i,0)];
            in->bee   [idx(i,-j-1)] = in->bee   [idx(i,0)];
            in->bex_re[idx(i,-j-1)] = in->bex_re[idx(i,0)];
            in->bex_im[idx(i,-j-1)] = in->bex_im[idx(i,0)];
            //upper side, estimated simply by extrapolation
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

            // 1) advection term:
            //   4-th order FD for 2nd-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor = -vz[i]/(12*dz);
            out->ee    [idx(i,j)] = factor * ((   ee[-2]-  ee[2]) - 8.0*(  ee[-1]-  ee[1]) );
            out->ex_re [idx(i,j)] = factor * ((  exr[-2]- exr[2]) - 8.0*( exr[-1]- exr[1]) );
            out->ex_im [idx(i,j)] = factor * ((  exi[-2]- exi[2]) - 8.0*( exi[-1]- exi[1]) );
            out->bee   [idx(i,j)] = factor * ((  bee[-2]- bee[2]) - 8.0*( bee[-1]- bee[1]) );
            out->bex_re[idx(i,j)] = factor * (( bexr[-2]-bexr[2]) - 8.0*(bexr[-1]-bexr[1]) );
            out->bex_im[idx(i,j)] = factor * (( bexi[-2]-bexi[2]) - 8.0*(bexi[-1]-bexi[1]) );

            // 2) prepare terms for -i [H0, rho]
            out->ee    [idx(i,j)] += -2*st*exi [0];
            out->ex_re [idx(i,j)] += -2*ct*exi [0];
            out->ex_im [idx(i,j)] +=  2*ct*exr [0] + st*( 2*ee[0] -1 );
            out->bee   [idx(i,j)] += -2*st*bexi[0];
            out->bex_re[idx(i,j)] += -2*ct*bexi[0];
            out->bex_im[idx(i,j)] +=  2*ct*bexr[0] + st*( 2*bee[0]-1 );

#ifndef MUINT_OFF
            // 3) interaction terms: vz-integral with a simple trapezoidal rule
            real Iee    = 0;
            real Iexr   = 0;
            real Iexi   = 0;
            real Ibee   = 0;
            real Ibexr  = 0;
            real Ibexi  = 0;
            for (int k=0;k<nvz; k++) {
                real eep    = (in->ee    [idx(k,j)]);
                real expr   = (in->ex_re [idx(k,j)]);
                real expi   = (in->ex_im [idx(k,j)]);
                real beep   = (in->bee   [idx(k,j)]);
                real bexpr  = (in->bex_re[idx(k,j)]);
                real bexpi  = (in->bex_im[idx(k,j)]);
		
		// terms for -i* mu * [rho'-rho_bar', rho]
		Iee   += 2*mu* (1-vz[i]*vz[k])*  (       exr[0] *(expi - bexpi) -    exi[0]*(expr- bexpr) );
                Iexr  +=   mu* (1-vz[i]*vz[k])*  (   (1-2*ee[0])*(expi - bexpi) +  2*exi[0]*(eep - beep ) );
                Iexi  +=   mu* (1-vz[i]*vz[k])*  (  -(1-2*ee[0])*(expr - bexpr) -  2*exr[0]*(eep - beep ) );
                Ibee  += 2*mu* (1-vz[i]*vz[k])*  (     -bexr[0] *(expi - bexpi) +   bexi[0]*(expr- bexpr) );
                Ibexr +=   mu* (1-vz[i]*vz[k])*  ( -(1-2*bee[0])*(expi - bexpi) - 2*bexi[0]*(eep - beep ) );
                Ibexi +=   mu* (1-vz[i]*vz[k])*  (  (1-2*bee[0])*(expr - bexpr) + 2*bexr[0]*(eep - beep ) );
            }
            // 3.1) calculate integral with simple trapezoidal rule
            out->ee    [idx(i,j)] += dv*(Iee   - 0.5*(vz[0]+vz[nvz-1])  );
            out->ex_re [idx(i,j)] += dv*(Iexr  - 0.5*(vz[0]+vz[nvz-1])  );
            out->ex_im [idx(i,j)] += dv*(Iexi  - 0.5*(vz[0]+vz[nvz-1])  );
            out->bee   [idx(i,j)] += dv*(Ibee  - 0.5*(vz[0]+vz[nvz-1])  );
            out->bex_re[idx(i,j)] += dv*(Ibexr - 0.5*(vz[0]+vz[nvz-1])  );
            out->bex_im[idx(i,j)] += dv*(Ibexi - 0.5*(vz[0]+vz[nvz-1])  );
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

    // avg, std, min, max
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

    // avg, std, min, max
    res[0] = vmin;
    res[1] = vmax;
    res[2] = sum/(nz*nvz);
    res[3] = sqrt( sum2/(nz*nvz) - res[2]*res[2]  );
}

void NuOsc::analysis() {

    real statis1[4], statis2[4];
    _analysis_c(statis1, v_stat-> ex_re, v_stat-> ex_im);
    _analysis_c(statis2, v_stat->bex_re, v_stat->bex_im);

    printf("Time: %10f exr:( %10g %10g %10g ) exi:( %10g %10g %10g)\n", phy_time, 
			statis1[0], statis1[1], statis1[3],
			statis2[0], statis2[1], statis2[3]);
			
    anafile << phy_time <<" "<< statis1[0]<< " " << statis1[1] << " "<< statis1[3] <<" "
                             << statis2[0]<< " " << statis2[1] << " "<< statis2[3] << endl;
}

void NuOsc::dumpv(const real v[]) {

    cout << "		=== ee component ===" << endl;
    for (int i=0;i<nvz; i++) {
        for (int j=-gz;j<nz+gz; j++) {
            printf("%10.2e ", v[idx(i,j)]);
        }
        cout << endl;
    }
    cout << endl;
}



int main(int argc, char *argv[]) {

    int END_TIME   = 5000;
    int DUMP_EVERY = 50000;
    int ANAL_EVERY = 100;
    
    int nz  = 256;
    int nvz = 128 + 1;

    for (int t = 1; argv[t] != 0; t++) {
    //	if (strcmp(argv[t], "--noint") == 0 )  state.nu_interaction = atoi(argv[t+1]);
	if (strcmp(argv[t], "--dim") == 0 )  {
	    nz  = atoi(argv[t+1]);
	    nvz = atoi(argv[t+2]);     t+=2;
	} else if (strcmp(argv[t], "--time") == 0 )  {
	    END_TIME   = atoi(argv[t+1]); t+=1;
	} else if (strcmp(argv[t], "--ana") == 0 )  {
	    DUMP_EVERY = atoi(argv[t+1]); t+=1;
	} else {
	    printf("Unreconganized parameters %s!\n", argv[t]);
	    exit(0);
	}
    }
    
    char fn[32];
    // Initialize simuation
    NuOsc state(nz, nvz);
    
    // initial value
    state.fillInitValue();
    state.write_bin("stat_init.bin");

    for (int t=1; t<END_TIME; t++) {

        state.step_rk4();

        if ( t%ANAL_EVERY==0) {
            state.analysis();
        }
        if ( t%DUMP_EVERY==0) {
            state.analysis();
            sprintf(fn,"stat_%04d.bin", t);
            state.write_bin(fn);
        }
    }


    printf("Simulation completed.\n");

    return 0;
}
