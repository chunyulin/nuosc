#include "nuosc_class.h"

// for init data
inline real eps_c(real eps0, real z, real z0, real sigma)        { return eps0*std::exp(-(z-z0)*(z-z0)/(2.0*sigma*sigma)); }
inline real eps_r(real eps0, real z=0, real z0=0, real sigma=0 ) { return eps0*rand()/RAND_MAX;}
inline real eps_p(real eps0, real z, real z0,  real sigma)       { return eps0*(1.0+cos(2*M_PI*(z-z0)/(2.0*sigma*sigma)))*0.5; }

double g(double vx, double vy, double vz, double s[], double v0 = 1.0) {
    return std::exp( - (vx-v0)*(vx-v0)/(2.0*s[0]*s[0]) - (vy-v0)*(vy-v0)/(2.0*s[1]*s[1]) - (vz-v0)*(vz-v0)/(2.0*s[2]*s[2]) );
}
double g(double vx, double vz, double sx, double sz, double vx0 = 1.0, double vz0 = 1.0) {
    return std::exp( - (vx-vx0)*(vx-vx0)/(2.0*sx*sx) - (vz-vz0)*(vz-vz0)/(2.0*sz*sz) );
}
double g(double v, double sigma, double v0 = 1.0){
    double N = sigma*std::sqrt(0.5*M_PI)*(std::erf((1.0+v0)/sigma/std::sqrt(2.0))+std::erf((1.0-v0)/sigma/std::sqrt(2.0)));
    //cout << "== Checking A : " << 1/N << endl;
    return std::exp( - (v-v0)*(v-v0)/(2.0*sigma*sigma) ) / N;
}

void NuOsc::fillInitValue(int ipt, real alpha, real eps0, real sigma, real lnue[], real lnueb[]) {
#ifdef PROFILE
    nvtxRangePush(__FUNCTION__);
#endif

    real n00=0, n01=0;

    if (ipt==4) {   // init data for the code comparison ptoject.

        int amax=nx[DIM-1]/2/10;

	if (myrank==0) printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g kmax=%d\n", "NOC paper", eps0, alpha, lnue[2], lnueb[2], sigma, amax);

        Vec phi(nx[DIM-1]/10+1);
        const real pi2oL = 2.0*M_PI/(bbox[DIM-1][1]-bbox[DIM-1][0]);
        for(int k=-amax;k<=amax;++k){
	    phi[k+amax]=2.0*M_PI*rand()/RAND_MAX;
        }
   
	#pragma omp parallel for reduction(+:n00,n01)
        for (int k=0;k<nx[DIM-1]; ++k){
	    real tmpr=0.0;
            real tmpi=0.0;
    	    for(int q=-amax;q<amax;++q) {
        	if(q!=0){
            	    tmpr += 1.e-7/abs(q)*cos(pi2oL*q*X[DIM-1][k] + phi[q+amax]);
        	    tmpi += 1.e-7/abs(q)*sin(pi2oL*q*X[DIM-1][k] + phi[q+amax]);
                }
            }
    	    real tmp2=sqrt(1.0-tmpr*tmpr-tmpi*tmpi);
    	    for (int v=0;v<nv; ++v){
		auto kv = idx(0,0,k,v);

		// ELN profile
		G0 [kv] =         g(vz[v], lnue [3]);
		G0b[kv] = alpha * g(vz[v], lnueb[3]);

                v_stat->wf[ff::ee]  [kv] =  0.5* G0 [kv]*(1.0+tmp2);//sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
                v_stat->wf[ff::mm]  [kv] =  0.5* G0 [kv]*(1.0-tmp2);
                v_stat->wf[ff::emr] [kv] =  0.5* G0 [kv]*tmpr;//1e-6;
                v_stat->wf[ff::emi] [kv] =  0.5* G0 [kv]*tmpi;//random_amp(0.001);
                v_stat->wf[ff::bee] [kv] =  0.5* G0b[kv]*(1.0+tmp2);
                v_stat->wf[ff::bmm] [kv] =  0.5* G0b[kv]*(1.0-tmp2);
                v_stat->wf[ff::bemr][kv] =  0.5* G0b[kv]*tmpr;//1e-6;
                v_stat->wf[ff::bemi][kv] = -0.5* G0b[kv]*tmpi;//random_amp(0.001);
                // initial nv_e
                n00 += vw[v]*v_stat->wf[ff::ee][kv];
                n01 += vw[v]*v_stat->wf[ff::bee][kv];
            }
        }

    } else {

	if (myrank==0) printf("   Init data: [%s] alpha= %f eps= %g sigma= %g lnu:[ %g %g %g ]  lnub:[ %g %g %g ]\n", ipt==0? "Point-like pertur":"Random pertur", alpha, eps0, sigma, lnue[0],lnue[1],lnue[2], lnueb[0],lnueb[1],lnueb[2] );

        // calulate normalization factor numerically...
	Vec ng(nv), ngb(nv);
	
        real ing0=0, ing1=0;
        #pragma omp parallel for simd reduction(+:ing0, ing1)
	for (int v=0;v<nv;++v) {
	    ng [v] = g(vx[v], vy[v], vz[v], lnue );
	    ngb[v] = g(vx[v], vy[v], vz[v], lnueb );
            ing0 += vw[v]*ng [v];
            ing1 += vw[v]*ngb[v];
	}

	ing0 = 1.0/ing0;    // for normalize G0, which means we don't need provide N actually.
	ing1 = 1.0/ing1;

        real (*spatialeps)(real,real,real,real);
        if      (ipt==0) { spatialeps = &eps_c; }      // center Z perturbation
        else if (ipt==1) { spatialeps = &eps_r; }      // random
        else if (ipt==2) { spatialeps = &eps_p; }      // periodic Z perturbation
        //else if (ipt==3) { spatialeps = 0;  }       // constant
        else             { assert(0); }                         // Not implemented

        #pragma omp parallel for reduction(+:n00,n01) collapse(COLLAPSE_LOOP)
        FORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);

            // ELN profile
            G0 [ijkv] =         ng [v] * ing0;
            G0b[ijkv] = alpha * ngb[v] * ing1;

            real tmpr = spatialeps(eps0, X[DIM-1][k], 0., sigma);
            real p3o = sqrt(1.0-tmpr*tmpr);
            v_stat->wf[ff::ee]  [ijkv] = 0.5* G0[ijkv]*(1.0+p3o);
            v_stat->wf[ff::mm]  [ijkv] = 0.5* G0[ijkv]*(1.0-p3o);
            v_stat->wf[ff::emr] [ijkv] = 0.5* G0[ijkv]*tmpr;
            v_stat->wf[ff::emi] [ijkv] = 0.0;
            v_stat->wf[ff::bee] [ijkv] = 0.5* G0b[ijkv]*(1.0+p3o);
            v_stat->wf[ff::bmm] [ijkv] = 0.5* G0b[ijkv]*(1.0-p3o);
            v_stat->wf[ff::bemr][ijkv] = 0.5* G0b[ijkv]*tmpr;
            v_stat->wf[ff::bemi][ijkv] = 0.0;
            #if NFLAVOR == 3
            v_stat->wf[ff::tt]  [ijkv] = 0.0;
            v_stat->wf[ff::mtr] [ijkv] = 0.0;// 0.5* G0[ijkv]*tmpr;
            v_stat->wf[ff::mti] [ijkv] = 0.0;
            v_stat->wf[ff::ter] [ijkv] = 0.0;// 0.5* G0[ijkv]*tmpr;
            v_stat->wf[ff::tei] [ijkv] = 0.0;
            v_stat->wf[ff::btt] [ijkv] = 0.0;
            v_stat->wf[ff::bmtr][ijkv] = 0.0;// 0.5* G0b[ijkv]*tmpr;
            v_stat->wf[ff::bmti][ijkv] = 0.0;
            v_stat->wf[ff::bter][ijkv] = 0.0;// 0.5* G0b[ijkv]*tmpr;
            v_stat->wf[ff::btei][ijkv] = 0.0;
            #endif

            // initial nv_e
            n00 += vw[v]*v_stat->wf[ff::ee ][ijkv];
            n01 += vw[v]*v_stat->wf[ff::bee][ijkv];
        }

        real n0[] = {n00, n01};
#ifdef COSENU_MPI
        MPI_Reduce(n0, n_nue0, 2, MPI_DOUBLE, MPI_SUM, 0, CartCOMM);
#else
        n_nue0[0] = n00;
        n_nue0[1] = n01;
#endif
    } //  end select case (ipt)

    n_nue0[0] *= ds_L;   // initial n_nue
    n_nue0[1] *= ds_L;   // initial n_nueb

    if (myrank==0) printf("      init number density of nu_e: %g %g\n", n_nue0[0], n_nue0[1]);

#ifdef PROFILE
    nvtxRangePop();
#endif
}

void NuOsc::fillInitGaussian(real eps0, real sigma) {

    if (myrank==0) printf("   Init Gaussian eps0= %g sigma= %g for testing.\n", eps0, sigma);

    PARFORALL(i,j,k,v) {
    
            auto ijkv = idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            real tmp = eps0* exp( - ((X[0][i])*(X[0][i]))/(1.0*sigma*sigma)
                                  - ((X[1][j])*(X[1][j]))/(1.0*sigma*sigma)
                                  - ((X[2][k])*(X[2][k]))/(1.0*sigma*sigma)  );
            v_stat->wf[ff::ee]  [ijkv] = tmp;
            v_stat->wf[ff::mm]  [ijkv] = 0;
            v_stat->wf[ff::emr] [ijkv] = 0;
            v_stat->wf[ff::emi] [ijkv] = 0;
            v_stat->wf[ff::bee] [ijkv] = 0;
            v_stat->wf[ff::bmm] [ijkv] = 0;
            v_stat->wf[ff::bemr][ijkv] = 0;
            v_stat->wf[ff::bemi][ijkv] = 0;
    }
}

void NuOsc::fillInitSquare(real eps0, real sigma) {

    if (myrank==0) printf("   Init Square eps0= %g sigma= %g for testing.\n", eps0, sigma);

        PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            real tmp = 0;
            if (X[2][j]*X[2][j]+X[0][i]*X[0][i] <= sigma*sigma) tmp = eps0;
            v_stat->wf[ff::ee]  [ijkv] = tmp;
            v_stat->wf[ff::mm]  [ijkv] = 0;
            v_stat->wf[ff::emr] [ijkv] = 0;
            v_stat->wf[ff::emi] [ijkv] = 0;
            v_stat->wf[ff::bee] [ijkv] = 0;
            v_stat->wf[ff::bmm] [ijkv] = 0;
            v_stat->wf[ff::bemr][ijkv] = 0;
            v_stat->wf[ff::bemi][ijkv] = 0;
        }
}

void NuOsc::fillInitTriangle(real eps0, real sigma) {

    printf("   Init Triangle eps0= %g sigma= %g for testing.\n", eps0, sigma);

    PARFORALL(i,j,k,v) {
            auto ijkv = idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            if      (X[DIM-1][k]<0  && X[DIM-1][k] > -sigma)  v_stat->wf[ff::ee][ijkv] = ( sigma + X[DIM-1][k] ) * eps0 / sigma;
            else if (X[DIM-1][k]>=0 && X[DIM-1][k] <  sigma)  v_stat->wf[ff::ee][ijkv] = ( sigma - X[DIM-1][k] ) * eps0 / sigma;
            else    v_stat->wf[ff::ee][ijkv] = 0.0;
            v_stat->wf[ff::mm]  [ijkv] = 0;
            v_stat->wf[ff::emr] [ijkv] = 0;
            v_stat->wf[ff::emi] [ijkv] = 0;
            v_stat->wf[ff::bee] [ijkv] = 0;
            v_stat->wf[ff::bmm] [ijkv] = 0;
            v_stat->wf[ff::bemr][ijkv] = 0;
            v_stat->wf[ff::bemi][ijkv] = 0;
    }
}
