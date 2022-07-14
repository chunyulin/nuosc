#include "nuosc_class.h"

// for init data
inline real eps_c(real z, real z0, real eps0, real sigma){return eps0*exp(-(z-z0)*(z-z0)/(2.0*sigma*sigma)); }
inline real eps_r(real eps0) {return eps0*rand()/RAND_MAX;}
double g(double v, double v0, double sigma){
    double N = sigma*std::sqrt(M_PI/2.0)*(erf((1.0+v0)/sigma/std::sqrt(2.0))+erf((1.0-v0)/sigma/std::sqrt(2.0)));
    //cout << "== Checking A : " << 1/N << endl;
    return exp( - (v-v0)*(v-v0)/(2.0*sigma*sigma) )/N;
}

// v quaduture on 2D unit disk
int gen_v2d_simple(const int nv, real *& vw, real *& vy, real *& vz) {
    real dv = 2.0/nv;
    vy = new real[nv*nv];
    vz = new real[nv*nv];
    vw = new real[nv*nv];
    uint co = 0;
    for (int i=0;i<nv; i++)
    for (int j=0;j<nv; j++) {
	real y = (0.5+i)*dv - 1;
        real z = (0.5+j)*dv - 1;
        if ((y*y+z*z) <= 1.0) {
    	    vy[co] = y;
    	    vz[co] = z;
            vw[co] = dv*dv;
            co++;
        }
    }
    return co;  // co < nv*nv
}

// v quaduture in [-1:1], vertex-center with simple trapezoidal rules.
int gen_v1d_trapezoidal(const int nv, real *& vw, real *& vz) {
    assert(nv%2==1);
    real dv = 2.0/(nv-1);
    vz = new real[nv];
    vw = new real[nv];
    for (int j=0;j<nv; j++) {
        vz[j] = j*dv - 1;
        vw[j] = dv;
    }
    vw[0]    = 0.5*dv;
    vw[nv-1] = 0.5*dv;
    return nv;
}

// v quaduture in [-1:1], vertex-center with Simpson 1/3 rule on uniform rgid.
int gen_v1d_simpson(const int nv, real *& vw, real *& vz) {
    assert(nv%2==1);
    real dv = 2.0/(nv-1);
    vz = new real[nv];
    vw = new real[nv];
    const real o3dv = 1./3.*dv;
    for (int j=0;j<nv; j++) {
        vz[j] = j*dv - 1;
        vw[j] = 2*((j%2)+1)*o3dv;
    }
    vw[0]    = o3dv;
    vw[nv-1] = o3dv;
    return nv;
}

int gen_v1d_cellcenter(const int nv, real *& vw, real *& vz) {
    assert(nv%2==0);
    real dv = 2.0/(nv);
    vz = new real[nv];
    vw = new real[nv];
    for (int j=0;j<nv; j++) {
        vz[j] = (j+0.5)*dv - 1;
        vw[j] = dv;
    }
    return nv;
}


void NuOsc::fillInitValue(int ipt, real alpha, real lnue, real lnueb,real eps0, real sigma) {

    real ne0 = 0.0, nb0 = 0.0;

    if (ipt==4) {

        int amax=nz/2/10;

	printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g kmax=%d\n", "NOC paper", eps0, alpha, lnue, lnueb, sigma, amax);

        std::vector<real> phi(nz/10+1);
        const real pi2oL = 2.0*M_PI/(z0-z1);
	#pragma omp parallel for
        for(int k=-amax;k<=amax;k++){
	    phi[k+amax]=2.0*M_PI*rand()/RAND_MAX;
        }
	#pragma omp parallel for reduction(+:ne0,nb0)
        for (int j=0;j<nz; j++){
	    real tmpr=0.0;
            real tmpi=0.0;
	    #pragma omp parallel for reduction(+:tmpr,tmpi),
    	    for(int k=-amax;k<amax;k++) {
        	if(k!=0){
            	    tmpr += 1.e-7/std::abs(k)*std::cos(pi2oL*k*Z[j] + phi[k+amax]);
        	    tmpi += 1.e-7/std::abs(k)*std::sin(pi2oL*k*Z[j] + phi[k+amax]);
                }
            }
    	    real tmp2=std::sqrt(1.0-tmpr*tmpr-tmpi*tmpi);
    	    for (int v=0;v<nv; v++){
		auto jv = idx(0,j,v);
		
		// ELN profile
		G0 [jv] =         g(vz[v], 1.0, lnue );
		G0b[jv] = alpha * g(vz[v], 1.0, lnueb);

            	v_stat->ee    [jv] =  0.5* G0 [jv]*(1.0+tmp2);//sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
                v_stat->xx    [jv] =  0.5* G0 [jv]*(1.0-tmp2);
                v_stat->ex_re [jv] =  0.5* G0 [jv]*tmpr;//1e-6;
                v_stat->ex_im [jv] =  0.5* G0 [jv]*tmpi;//random_amp(0.001);
                v_stat->bee   [jv] =  0.5* G0b[jv]*(1.0+tmp2);
                v_stat->bxx   [jv] =  0.5* G0b[jv]*(1.0-tmp2);
                v_stat->bex_re[jv] =  0.5* G0b[jv]*tmpr;//1e-6;
                v_stat->bex_im[jv] = -0.5* G0b[jv]*tmpi;//random_amp(0.001);
                // initial nv_e
                ne0 += vw[v]*v_stat->ee [jv];
                nb0 += vw[v]*v_stat->bee[jv];
            }
        }

    } else {
    
	printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g\n", ipt==0? "Point-like pertur":"Random pertur", eps0, alpha, lnue, lnueb, sigma);

	#pragma omp parallel for reduction(+:ne0,nb0) collapse(COLLAPSE_LOOP)
        FORALL(i,j,v) {
    
	    auto ijv = idx(i,j,v);
	    
	    // ELN profile
            G0 [ijv] =         g(vz[v], 1.0, lnue );
            G0b[ijv] = alpha * g(vz[v], 1.0, lnueb);

            real tmpr;
            if      (ipt==0) { tmpr = eps_c(Z[j],0.0,eps0,sigma); }   // center perturbation
            else if (ipt==1) { tmpr = eps_r(eps0); }                 // random
            else             { assert(0); }                         // Not implemented

            real p3o = std::sqrt(1.0-tmpr*tmpr);
            v_stat->ee    [ijv] = 0.5* G0[ijv]*(1.0+p3o);
            v_stat->xx    [ijv] = 0.5* G0[ijv]*(1.0-p3o);
            v_stat->ex_re [ijv] = 0.5* G0[ijv]*tmpr;
            v_stat->ex_im [ijv] = 0.0;
            v_stat->bee   [ijv] = 0.5* G0b[ijv]*(1.0+p3o);
            v_stat->bxx   [ijv] = 0.5* G0b[ijv]*(1.0-p3o);
            v_stat->bex_re[ijv] = 0.5* G0b[ijv]*tmpr;
            v_stat->bex_im[ijv] = 0.0;

            // initial nv_e
            ne0 += vw[v]*v_stat->ee [ijv];
            nb0 += vw[v]*v_stat->bee[ijv];
	}
    
    } //  end select case (ipt)

    n_nue0  = ne0*dz/(z1-z0);   // initial n_nue
    n_nueb0 = nb0*dz/(z1-z0);   // initial n_nueb

    printf("      init number density of nu_e: %g %g\n", n_nue0, n_nueb0);
    
#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif


#if 0
    // dumpG
    std::ofstream o;
    char fn[32];
    sprintf(fn, "G0_%f.dat", alpha);
    o.open(fn, std::ofstream::out | std::ofstream::trunc);
    for (int v=0;v<nv;v++) {
	    auto ijv = idx(1,1,v);
	    o << vz[v] << " " << G0[ijv] - G0b[ijv] << endl;
    }
    o.close();
#endif
#if 0
    // dumpP1
    std::ofstream o;
    char fn[32];
    sprintf(fn, "P1_%f.dat", alpha);
    o.open(fn, std::ofstream::out | std::ofstream::trunc);
    for (int v=0;v<nv;v++) {
	    auto ijv = idx(1,1,v);
	    o << vz[v] << " " << G0[ijv] - G0b[ijv] << endl;
    }
    o.close();
#endif

}


void NuOsc::fillInitGaussian(real eps0, real sigma) {

    printf("   Init Gaussian eps0= %g sigma= %g for testing.", eps0, sigma);

    PARFORALL(i,j,v) {
    
            auto ipv = idx(i,j,v);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;

            real tmp = eps0* std::exp( - Z[j]*Z[j]/(2.0*sigma*sigma) );    // Initial P3 in our case
            v_stat->ee    [ipv] = tmp;
            v_stat->xx    [ipv] = 0;
            v_stat->ex_re [ipv] = 0;
            v_stat->ex_im [ipv] = 0;
            v_stat->bee   [ipv] = 0;
            v_stat->bxx   [ipv] = 0;
            v_stat->bex_re[ipv] = 0;
            v_stat->bex_im[ipv] = 0;
    }
}

void NuOsc::fillInitSquare(real eps0, real sigma) {

    printf("   Init Square eps0= %g sigma= %g for testing.", eps0, sigma);

    PARFORALL(i,j,v) {
    
	    auto ipv = idx(i,j,v);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;
	    
            real tmp = 0;
            if (Z[j]*Z[j]<sigma*sigma) tmp = eps0;
            v_stat->ee    [ipv] = tmp;
            v_stat->xx    [ipv] = 0;
            v_stat->ex_re [ipv] = 0;
            v_stat->ex_im [ipv] = 0;
            v_stat->bee   [ipv] = 0;
            v_stat->bxx   [ipv] = 0;
            v_stat->bex_re[ipv] = 0;
            v_stat->bex_im[ipv] = 0;
    }
}

void NuOsc::fillInitTriangle(real eps0, real sigma) {

    printf("   Init Triangle eps0= %g sigma= %g for testing.", eps0, sigma);

    PARFORALL(i,j,v) {
	    auto ipv = idx(i,j,v);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;
	    
            if      (Z[j]<0  && Z[j] > -sigma)  v_stat->ee[ipv] = ( sigma + Z[j] ) * eps0 / sigma;
            else if (Z[j]>=0 && Z[j] <  sigma)  v_stat->ee[ipv] = ( sigma - Z[j] ) * eps0 / sigma;
            else                                    v_stat->ee[ipv] = 0.0;
            v_stat->xx    [ipv] = 0;
            v_stat->ex_re [ipv] = 0;
            v_stat->ex_im [ipv] = 0;
            v_stat->bee   [ipv] = 0;
            v_stat->bxx   [ipv] = 0;
            v_stat->bex_re[ipv] = 0;
            v_stat->bex_im[ipv] = 0;
    }
}

