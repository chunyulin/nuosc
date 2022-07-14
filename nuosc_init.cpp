#include "nuosc_class.h"

// for init data
inline real eps_c(real z, real z0, real eps0, real sigma){return eps0*std::exp(-(z-z0)*(z-z0)/(2.0*sigma*sigma)); }
inline real eps_r(real eps0) {return eps0*rand()/RAND_MAX;}
double g(double v, double v0, double sigma) {
    double N = sigma*std::sqrt(M_PI/2.0)*(std::erf((1.0+v0)/sigma/std::sqrt(2.0))+std::erf((1.0-v0)/sigma/std::sqrt(2.0)));
    return std::exp( - (v-v0)*(v-v0)/(2.0*sigma*sigma) )/N;
}

// 1D vertex-center v-grid in [-1:1]
int gen_v1d_vertex_center(const int nv, real *& vw, real *& vz) {
    assert(nv%2==1);  // just for convenience to include v=1, -1, 0 via trapezoidal rule
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
// 1D vertex-center v-grid in [-1:1]
int gen_v1d_cell_center(const int nv, real *& vw, real *& vz) {
    assert(nv%2==0);  // just for convenience to include v=1, -1, 0 via trapezoidal rule
    real dv = 2.0/nv;
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
        for (int i=0;i<nz; ++i)
        for (int p=0;p<np; ++p) {

            auto ip  = idx(i,p);

            real tmpr=0.0;
            real tmpi=0.0;
            #pragma omp parallel for reduction(+:tmpr,tmpi),
            for(int k=-amax;k<amax;k++) {
                if(k!=0){
                    tmpr += 1.e-7/std::abs(k)*std::cos(pi2oL*k*Z[ip] + phi[k+amax]);
                    tmpi += 1.e-7/std::abs(k)*std::sin(pi2oL*k*Z[ip] + phi[k+amax]);
                }
            }
            real tmp2=std::sqrt(1.0-tmpr*tmpr-tmpi*tmpi);
            for (int v=0;v<nv; v++){
                auto ipv = idx(i,p,v);
                // ELN profile
                G0 [ipv] =         g(vz[v], 1.0, lnue );
                G0b[ipv] = alpha * g(vz[v], 1.0, lnueb);

                v_stat->ee    [ipv] =  0.5* G0 [ipv]*(1.0+tmp2);//sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
                v_stat->xx    [ipv] =  0.5* G0 [ipv]*(1.0-tmp2);
                v_stat->ex_re [ipv] =  0.5* G0 [ipv]*tmpr;//1e-6;
                v_stat->ex_im [ipv] =  0.5* G0 [ipv]*tmpi;//random_amp(0.001);
                v_stat->bee   [ipv] =  0.5* G0b[ipv]*(1.0+tmp2);
                v_stat->bxx   [ipv] =  0.5* G0b[ipv]*(1.0-tmp2);
                v_stat->bex_re[ipv] =  0.5* G0b[ipv]*tmpr;//1e-6;
                v_stat->bex_im[ipv] = -0.5* G0b[ipv]*tmpi;//random_amp(0.001);
                // initial nv_e
                ne0 += vw[v]*v_stat->ee [ipv]*w[p];
                nb0 += vw[v]*v_stat->bee[ipv]*w[p];
            }
        }

    } else {    // for ipt=0,1,2
        
        printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g\n", ipt==0? "Point-like pertur":"Random pertur", eps0, alpha, lnue, lnueb, sigma);

        for (int i=0;i<nz; ++i)
        for (int p=0;p<np; ++p)
        for (int v=0;v<nv; ++v)  {

            auto ipv = idx(i,p,v);
            auto ip  = idx(i,p);

            // ELN profile
            G0 [ipv] =         g(vz[v], 1.0, lnue );
            G0b[ipv] = alpha * g(vz[v], 1.0, lnueb);

            real tmp;
            if      (ipt==0) { tmp = eps_c(Z[ip],0.0,eps0,sigma); }  // center perturbation
            else if (ipt==1) { tmp = eps_r(eps0); }                 // random
#ifdef COSENU2D
            else if (ipt==2) { tmp = eps_c(Y[ip],0.0,eps0,sigma)*eps_c(Z[ip],0.0,eps0,sigma); }  // 2D symmetric gaussian
#endif
            else             { assert(0 && "Not implemented ipt."); }

            real p3o = sqrt(1.0-tmp*tmp);    // Initial P3 in our case
            v_stat->ee    [ipv] = 0.5* G0[ipv]*(1.0+p3o);
            v_stat->xx    [ipv] = 0.5* G0[ipv]*(1.0-p3o);
            v_stat->ex_re [ipv] = 0.5* G0[ipv]*tmp;
            v_stat->ex_im [ipv] = 0.0;
            v_stat->bee   [ipv] = 0.5* G0b[ipv]*(1.0+p3o);
            v_stat->bxx   [ipv] = 0.5* G0b[ipv]*(1.0-p3o);
            v_stat->bex_re[ipv] = 0.5* G0b[ipv]*tmp;
            v_stat->bex_im[ipv] = 0.0;

            // initial nv_e
            ne0 += vw[v]*v_stat->ee [ipv]*w[p];
            nb0 += vw[v]*v_stat->bee[ipv]*w[p];
        }
    } //  end select case (ipt)
    
    n_nue0  = ne0*dz*0.5/(z1-z0);   // initial n_nue
    n_nueb0 = nb0*dz*0.5/(z1-z0);   // initial n_nueb
    printf("      init number density of nu_e: %g %g\n", n_nue0, n_nueb0);

/*
#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif
*/

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

    for (int i=0;i<nz; ++i)
    for (int p=0;p<np; ++p)
    for (int v=0;v<nv; ++v)  {
    
	    auto ipv = idx(i,p,v);
	    auto ip  = idx(i,p);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;
	    
            real tmp = eps0* std::exp( - Z[ip]*Z[ip]/(2.0*sigma*sigma) );    // Initial P3 in our case
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

    for (int i=0;i<nz; ++i)
    for (int p=0;p<np; ++p)
    for (int v=0;v<nv; ++v)  {
    
	    auto ipv = idx(i,p,v);
	    auto ip  = idx(i,p);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;
	    
            real tmp = 0;
            if (Z[ip]*Z[ip]<sigma*sigma) tmp = eps0;
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

    for (int i=0;i<nz; ++i)
    for (int p=0;p<np; ++p)
    for (int v=0;v<nv; ++v)  {
    
	    auto ipv = idx(i,p,v);
	    auto ip  = idx(i,p);

            G0 [ipv] = 1.0;
            G0b[ipv] = 1.0;
	    
            if      (Z[ip]<0  && Z[ip] > -sigma)  v_stat->ee[ipv] = ( sigma + Z[ip] ) * eps0 / sigma;
            else if (Z[ip]>=0 && Z[ip] <  sigma)  v_stat->ee[ipv] = ( sigma - Z[ip] ) * eps0 / sigma;
            else                                  v_stat->ee[ipv] = 0.0;
            v_stat->xx    [ipv] = 0;
            v_stat->ex_re [ipv] = 0;
            v_stat->ex_im [ipv] = 0;
            v_stat->bee   [ipv] = 0;
            v_stat->bxx   [ipv] = 0;
            v_stat->bex_re[ipv] = 0;
            v_stat->bex_im[ipv] = 0;
    }
}

