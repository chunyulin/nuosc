#include "nuosc_class.h"

// for init data
inline real eps_c(real z, real z0, real eps0, real sigma){return eps0*std::exp(-(z-z0)*(z-z0)/(2.0*sigma*sigma)); }
inline real eps_r(real eps0) {return eps0*rand()/RAND_MAX;}
inline real eps_p(real z, real z0, real eps0, real sigma){return eps0*(1.0+cos(2*M_PI*(z-z0)/(2.0*sigma*sigma)))*0.5; }

double g(double vx, double vz, double sx, double sz, double vx0 = 1.0, double vz0 = 1.0) {
    // ELN for vx,vz
    return std::exp( - (vx-vx0)*(vx-vx0)/(2.0*sx*sx) - (vz-vz0)*(vz-vz0)/(2.0*sz*sz) );
}
double g(double v, double sigma, double v0 = 1.0){
    double N = sigma*std::sqrt(0.5*M_PI)*(std::erf((1.0+v0)/sigma/std::sqrt(2.0))+std::erf((1.0-v0)/sigma/std::sqrt(2.0)));
    //cout << "== Checking A : " << 1/N << endl;
    return std::exp( - (v-v0)*(v-v0)/(2.0*sigma*sigma) ) / N;
}

void NuOsc::fillInitValue(int ipt, real alpha, real eps0, real sigma, real lnue, real lnueb, real lnue_x, real lnueb_x) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    real n00=0, n01=0;

    if (ipt==4) {   // init data for the code comparison ptoject.

        int amax=grid.nx[DIM-1]/2/10;

	if (myrank==0) printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g kmax=%d\n", "NOC paper", eps0, alpha, lnue, lnueb, sigma, amax);

        Vec phi(grid.nx[DIM-1]/10+1);
        const real pi2oL = 2.0*M_PI/(grid.bbox[DIM-1][1]-grid.bbox[DIM-1][0]);
        for(int k=-amax;k<=amax;++k){
	    phi[k+amax]=2.0*M_PI*rand()/RAND_MAX;
        }
   
	#pragma omp parallel for reduction(+:n00,n01)
        for (int k=0;k<grid.nx[DIM-1]; ++k){
	    real tmpr=0.0;
            real tmpi=0.0;
    	    for(int q=-amax;q<amax;++q) {
        	if(q!=0){
            	    tmpr += 1.e-7/abs(q)*cos(pi2oL*q*grid.X[DIM-1][k] + phi[q+amax]);
        	    tmpi += 1.e-7/abs(q)*sin(pi2oL*q*grid.X[DIM-1][k] + phi[q+amax]);
                }
            }
    	    real tmp2=sqrt(1.0-tmpr*tmpr-tmpi*tmpi);
    	    for (int v=0;v<grid.nv; ++v){
		auto kv = grid.idx(0,0,k,v);

		// ELN profile
		G0 [kv] =         g(grid.vz[v], lnue );
		G0b[kv] = alpha * g(grid.vz[v], lnueb);

                v_stat->ee    [kv] =  0.5* G0 [kv]*(1.0+tmp2);//sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
                v_stat->xx    [kv] =  0.5* G0 [kv]*(1.0-tmp2);
                v_stat->ex_re [kv] =  0.5* G0 [kv]*tmpr;//1e-6;
                v_stat->ex_im [kv] =  0.5* G0 [kv]*tmpi;//random_amp(0.001);
                v_stat->bee   [kv] =  0.5* G0b[kv]*(1.0+tmp2);
                v_stat->bxx   [kv] =  0.5* G0b[kv]*(1.0-tmp2);
                v_stat->bex_re[kv] =  0.5* G0b[kv]*tmpr;//1e-6;
                v_stat->bex_im[kv] = -0.5* G0b[kv]*tmpi;//random_amp(0.001);
                // initial nv_e
                n00 += grid.vw[v]*v_stat->ee [kv];
                n01 += grid.vw[v]*v_stat->bee[kv];
            }
        }

    } else {

	if (myrank==0) printf("   Init data: [%s] alpha= %f eps= %g sigma= %g lnu_z= %g %g lnu_x= %g %g\n", ipt==0? "Point-like pertur":"Random pertur", alpha, eps0, sigma, lnue, lnueb, lnue_x, lnueb_x);

        // calulate normalization factor numerically...
	Vec ng(grid.nv), ngb(grid.nv);
	
        real ing0=0, ing1=0;
        #pragma omp parallel for reduction(+:ing0, ing1)
	for (int v=0;v<grid.nv;++v) {
	    ng [v] = g(grid.vx[v], grid.vz[v], lnue_x,  lnue );   // large vx sigma to reduce to 1D case (axi-symmetric case)
	    ngb[v] = g(grid.vx[v], grid.vz[v], lnueb_x, lnueb );
            ing0 += grid.vw[v]*ng [v];
            ing1 += grid.vw[v]*ngb[v];
	}

	ing0 = 1.0/ing0;
	ing1 = 1.0/ing1;

	#pragma omp parallel for reduction(+:n00,n01) collapse(COLLAPSE_LOOP)
        FORALL(i,j,k,v) {

            auto ijkv = grid.idx(i,j,k,v);

            // ELN profile
            G0 [ijkv] =         ng [v] * ing0;
            G0b[ijkv] = alpha * ngb[v] * ing1;

            real tmpr;
            if      (ipt==0) { tmpr = eps_c(grid.X[DIM-1][k],0.0,eps0,sigma); }      // center Z perturbation
            else if (ipt==1) { tmpr = eps_r(eps0); }                                 // random
            else if (ipt==2) { tmpr = eps_p(grid.X[DIM-1][k],0.0,eps0,sigma);}       // periodic Z perturbation
            else if (ipt==3) { tmpr = eps0;}
            else             { assert(0); }                         // Not implemented

            real p3o = sqrt(1.0-tmpr*tmpr);
            v_stat->ee    [ijkv] = 0.5* G0[ijkv]*(1.0+p3o);
            v_stat->xx    [ijkv] = 0.5* G0[ijkv]*(1.0-p3o);
            v_stat->ex_re [ijkv] = 0.5* G0[ijkv]*tmpr;
            v_stat->ex_im [ijkv] = 0.0;
            v_stat->bee   [ijkv] = 0.5* G0b[ijkv]*(1.0+p3o);
            v_stat->bxx   [ijkv] = 0.5* G0b[ijkv]*(1.0-p3o);
            v_stat->bex_re[ijkv] = 0.5* G0b[ijkv]*tmpr;
            v_stat->bex_im[ijkv] = 0.0;

            // initial nv_e
            n00 += grid.vw[v]*v_stat->ee [ijkv];
            n01 += grid.vw[v]*v_stat->bee[ijkv];
        }

        real n0[] = {n00, n01};
#ifdef COSENU_MPI
        MPI_Reduce(n0, n_nue0, 2, MPI_DOUBLE, MPI_SUM, 0, grid.CartCOMM);
#else
        n_nue0[0] = n00;
        n_nue0[1] = n01;
#endif
    } //  end select case (ipt)

    n_nue0[0] *= ds_L;   // initial n_nue
    n_nue0[1] *= ds_L;   // initial n_nueb

    if (myrank==0) printf("      init number density of nu_e: %g %g\n", n_nue0[0], n_nue0[1]);

#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::fillInitGaussian(real eps0, real sigma) {

    if (myrank==0) printf("   Init Gaussian eps0= %g sigma= %g for testing.\n", eps0, sigma);

    PARFORALL(i,j,k,v) {
    
            auto ijkv = grid.idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            real tmp = eps0* exp( - ((grid.X[0][i])*(grid.X[0][i]))/(1.0*sigma*sigma)
                                  - ((grid.X[2][k])*(grid.X[2][k]))/(1.0*sigma*sigma)  );
            v_stat->ee    [ijkv] = tmp;
            v_stat->xx    [ijkv] = 0;
            v_stat->ex_re [ijkv] = 0;
            v_stat->ex_im [ijkv] = 0;
            v_stat->bee   [ijkv] = 0;
            v_stat->bxx   [ijkv] = 0;
            v_stat->bex_re[ijkv] = 0;
            v_stat->bex_im[ijkv] = 0;
    }
}

void NuOsc::fillInitSquare(real eps0, real sigma) {

    if (myrank==0) printf("   Init Square eps0= %g sigma= %g for testing.\n", eps0, sigma);

    PARFORALL(i,j,k,v) {
	    auto ijkv = grid.idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            real tmp = 0;
            if (grid.X[DIM-1][k]*grid.X[DIM-1][k]+grid.X[0][i]*grid.X[0][i] <= sigma*sigma) tmp = eps0;
            v_stat->ee    [ijkv] = tmp;
            v_stat->xx    [ijkv] = 0;
            v_stat->ex_re [ijkv] = 0;
            v_stat->ex_im [ijkv] = 0;
            v_stat->bee   [ijkv] = 0;
            v_stat->bxx   [ijkv] = 0;
            v_stat->bex_re[ijkv] = 0;
            v_stat->bex_im[ijkv] = 0;
    }
}

void NuOsc::fillInitTriangle(real eps0, real sigma) {

    printf("   Init Triangle eps0= %g sigma= %g for testing.\n", eps0, sigma);

    PARFORALL(i,j,k,v) {
	    auto ijkv = grid.idx(i,j,k,v);

            G0 [ijkv] = 1.0;
            G0b[ijkv] = 1.0;

            if      (grid.X[DIM-1][k]<0  && grid.X[DIM-1][k] > -sigma)  v_stat->ee[ijkv] = ( sigma + grid.X[DIM-1][k] ) * eps0 / sigma;
            else if (grid.X[DIM-1][k]>=0 && grid.X[DIM-1][k] <  sigma)  v_stat->ee[ijkv] = ( sigma - grid.X[DIM-1][k] ) * eps0 / sigma;
            else    v_stat->ee[ijkv] = 0.0;
            v_stat->xx    [ijkv] = 0;
            v_stat->ex_re [ijkv] = 0;
            v_stat->ex_im [ijkv] = 0;
            v_stat->bee   [ijkv] = 0;
            v_stat->bxx   [ijkv] = 0;
            v_stat->bex_re[ijkv] = 0;
            v_stat->bex_im[ijkv] = 0;
    }
}
