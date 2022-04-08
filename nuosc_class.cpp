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
    const real o3dv = 1./3.*dv;
    vz = new real[nv];
    vw = new real[nv];
    for (int j=0;j<nv; j++) {
        vz[j] = j*dv - 1;
        vw[j] = 2*((j%2)+1)*o3dv;
    }
    vw[0]    = o3dv;
    vw[nv-1] = o3dv;
    return nv;
}


void NuOsc::fillInitValue(int ipt, real alpha, real lnue, real lnueb,real eps0, real sigma) {

    printf("   Init data: [%s] eps= %g  alpha= %f  sigma= %g %g  width= %g\n", ipt==0? "Point-like pertur":"Random pertur", eps0, alpha, lnue, lnueb, sigma);

    real ne0 = 0.0, nb0 = 0.0;

#pragma omp parallel for reduction(+:ne0,nb0) collapse(COLLAPSE_LOOP)
    FORALL(i,j,v) {
    
	    auto ijv = idx(i,j,v);
	    
	    // ELN profile
            G0 [ijv] =         g(vz[v], 1.0, lnue );
            G0b[ijv] = alpha * g(vz[v], 1.0, lnueb);

            real tmp;
            if      (ipt==0) { tmp = eps_c(Z[j],0.0,eps0,sigma); }   // center perturbation
            else if (ipt==1) { tmp = eps_r(eps0); }                 // random
            else             { assert(0); }                         // Not implemented

            real p3o = std::sqrt(1.0-tmp*tmp);
            v_stat->ee    [ijv] = 0.5* G0[ijv]*(1.0+p3o);
            v_stat->xx    [ijv] = 0.5* G0[ijv]*(1.0-p3o);
            v_stat->ex_re [ijv] = 0.5* G0[ijv]*tmp;
            v_stat->ex_im [ijv] = 0.0;
            v_stat->bee   [ijv] = 0.5* G0b[ijv]*(1.0+p3o);
            v_stat->bxx   [ijv] = 0.5* G0b[ijv]*(1.0-p3o);
            v_stat->bex_re[ijv] = 0.5* G0b[ijv]*tmp;
            v_stat->bex_im[ijv] = 0.0;

            // initial nv_e
            ne0 += vw[v]*v_stat->ee [ijv];
            nb0 += vw[v]*v_stat->bee[ijv];
    }
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

void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
#ifdef COSENU2D
    for (int i=0;i<ny; i++)
#else
    for (int i=0;i<1; i++)
#endif
    for (int j=0;j<gz; j++)
    for (int v=0;v<nv; v++) {
                //z lower side
                in->ee    [idx(i,-j-1,v)] = in->ee    [idx(i,nz-j-1,v)];
                in->xx    [idx(i,-j-1,v)] = in->xx    [idx(i,nz-j-1,v)];
                in->ex_re [idx(i,-j-1,v)] = in->ex_re [idx(i,nz-j-1,v)];
                in->ex_im [idx(i,-j-1,v)] = in->ex_im [idx(i,nz-j-1,v)];
                in->bee   [idx(i,-j-1,v)] = in->bee   [idx(i,nz-j-1,v)];
                in->bxx   [idx(i,-j-1,v)] = in->bxx   [idx(i,nz-j-1,v)];
                in->bex_re[idx(i,-j-1,v)] = in->bex_re[idx(i,nz-j-1,v)];
                in->bex_im[idx(i,-j-1,v)] = in->bex_im[idx(i,nz-j-1,v)];
                //z upper side
                in->ee    [idx(i,nz+j,v)] = in->ee    [idx(i,j,v)];
                in->xx    [idx(i,nz+j,v)] = in->xx    [idx(i,j,v)];
                in->ex_re [idx(i,nz+j,v)] = in->ex_re [idx(i,j,v)];
                in->ex_im [idx(i,nz+j,v)] = in->ex_im [idx(i,j,v)];
                in->bee   [idx(i,nz+j,v)] = in->bee   [idx(i,j,v)];
                in->bxx   [idx(i,nz+j,v)] = in->bxx   [idx(i,j,v)];
                in->bex_re[idx(i,nz+j,v)] = in->bex_re[idx(i,j,v)];
                in->bex_im[idx(i,nz+j,v)] = in->bex_im[idx(i,j,v)];
    }


#ifdef COSENU2D
#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
    for (int i=0;i<gy; i++)
    for (int j=0;j<nz; j++)
    for (int v=0;v<nv; v++) {
                //y lower side
                in->ee    [idx(-i-1,j,v)] = in->ee    [idx(ny-i-1,j,v)];
                in->xx    [idx(-i-1,j,v)] = in->xx    [idx(ny-i-1,j,v)];
                in->ex_re [idx(-i-1,j,v)] = in->ex_re [idx(ny-i-1,j,v)];
                in->ex_im [idx(-i-1,j,v)] = in->ex_im [idx(ny-i-1,j,v)];
                in->bee   [idx(-i-1,j,v)] = in->bee   [idx(ny-i-1,j,v)];
                in->bxx   [idx(-i-1,j,v)] = in->bxx   [idx(ny-i-1,j,v)];
                in->bex_re[idx(-i-1,j,v)] = in->bex_re[idx(ny-i-1,j,v)];
                in->bex_im[idx(-i-1,j,v)] = in->bex_im[idx(ny-i-1,j,v)];
                //y upper side
                in->ee    [idx(ny+i,j,v)] = in->ee    [idx(i,j,v)];
                in->xx    [idx(ny+i,j,v)] = in->xx    [idx(i,j,v)];
                in->ex_re [idx(ny+i,j,v)] = in->ex_re [idx(i,j,v)];
                in->ex_im [idx(ny+i,j,v)] = in->ex_im [idx(i,j,v)];
                in->bee   [idx(ny+i,j,v)] = in->bee   [idx(i,j,v)];
                in->bxx   [idx(ny+i,j,v)] = in->bxx   [idx(i,j,v)];
                in->bex_re[idx(ny+i,j,v)] = in->bex_re[idx(i,j,v)];
                in->bex_im[idx(ny+i,j,v)] = in->bex_im[idx(i,j,v)];
            }
#endif

#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { 
    cout << "Not implemented." << endl;
    assert(0);
}


void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    const int nzv = nv*nz;

#pragma omp parallel for collapse(3)
#ifdef COSENU2D
    for (int i=0;i<ny; i++)
#else
    for (int i=0;i<1; i++)
#endif
#pragma acc parallel loop gang
    for (int j=0;j<nz; j++)
#pragma acc loop vector
        for (int v=0;v<nv; v++) {

            // The base pointer for this stencil
            real *ee    = &(in->ee    [idx(i,j,v)]);
            real *xx    = &(in->xx    [idx(i,j,v)]);
            real *exr   = &(in->ex_re [idx(i,j,v)]);
            real *exi   = &(in->ex_im [idx(i,j,v)]);
            real *bee   = &(in->bee   [idx(i,j,v)]);
            real *bxx   = &(in->bxx   [idx(i,j,v)]);
            real *bexr  = &(in->bex_re[idx(i,j,v)]);
            real *bexi  = &(in->bex_im[idx(i,j,v)]);

            // prepare KO operator
#ifndef KO_ORD_3
            // Kreiss-Oliger dissipation (5-th order)
            real ko_eps_z = -ko/dz/64.0;
  #ifdef COSENU2D
            real ko_eps_y = -ko/dy/64.0;
            #define KO_FD(x) ( ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] ) + \
                               ko_eps_y*( x[-3*nzv] + x[3*nzv] - 6*(x[-2*nzv]+x[2*nzv]) + 15*(x[-nzv]+x[nzv]) - 20*x[0] ) )
  #else
            #define KO_FD(x)   ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] )
  #endif
#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps_z = -ko/dz/16.0;
  #ifdef COSENU2D
            real ko_eps_y = -ko/dy/16.0;
            #define KO_FD(x) ( ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] ) + \
                               ko_eps_y * ( x[-2*nzv] + x[2*nzv] - 4*(x[-nzv]+x[nzv]) + 6*x[0] ) )
  #else
	    #define KO_FD(x)   ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] )
  #endif
#endif

            // prepare advection FD operator
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor_z = -vz[v]/(12*dz);
#ifdef COSENU2D
            real factor_y = -vy[v]/(12*dy);
            #define ADV_FD(x) ( factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv]  ) ) + \
                                  factor_y*(  (x[-2*nzv]-x[2*nzv]) - 8.0*(x[-nzv]-x[nzv] ) ) )
#else
            #define ADV_FD(x)     factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv])  )
#endif
            // prepare vz-integral with a cubature rule
            real Iee    = 0;
            real Iexr   = 0;
            real Iexi   = 0;
            real Ibee   = 0;
            real Ibexr  = 0;
            real Ibexi  = 0;
#pragma acc loop
            for (int k=0;k<nv; k++) {   // vz' integral
                real eep    = (in->ee    [idx(i,j,k)]);
                real xxp    = (in->xx    [idx(i,j,k)]);
                real expr   = (in->ex_re [idx(i,j,k)]);
                real expi   = (in->ex_im [idx(i,j,k)]);
                real beep   = (in->bee   [idx(i,j,k)]);
                real bxxp   = (in->bxx   [idx(i,j,k)]);
                real bexpr  = (in->bex_re[idx(i,j,k)]);
                real bexpi  = (in->bex_im[idx(i,j,k)]);

                // terms for -i* mu * [rho'-rho_bar', rho]
                #ifdef COSENU2D
		real fvdv = vw[k]*mu* (1-vy[v]*vy[k]-vz[v]*vz[k]);
                #else
                real fvdv = vw[k]*mu* (1-vz[v]*vz[k]);
                #endif
                Iee   +=  2* fvdv * (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Iexr  +=     fvdv * (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=     fvdv * ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  2* fvdv * (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibexr +=     fvdv * ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=     fvdv * ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );

            }

#if 0
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [idx(i,j,v)] =  Iee   + ADV_FD(ee)   + KO_FD(ee)   - pmo* 2*st*exi [0];
            out->xx    [idx(i,j,v)] = -Iee   + ADV_FD(xx)   + KO_FD(xx)   + pmo* 2*st*exi [0];
            out->ex_re [idx(i,j,v)] =  Iexr  + ADV_FD(exr)  + KO_FD(exr)  - pmo* 2*ct*exi [0];
            out->ex_im [idx(i,j,v)] =  Iexi  + ADV_FD(exi)  + KO_FD(exi)  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [idx(i,j,v)] =  Ibee  + ADV_FD(bee)  + KO_FD(bee)  - pmo* 2*st*bexi[0];
            out->bxx   [idx(i,j,v)] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx)  + pmo* 2*st*bexi[0];
            out->bex_re[idx(i,j,v)] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr) - pmo* 2*ct*bexi[0];
            out->bex_im[idx(i,j,v)] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi) + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );
#else
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [idx(i,j,v)] =  Iee   + ADV_FD(ee)   + KO_FD(ee);
            out->xx    [idx(i,j,v)] = -Iee   + ADV_FD(xx)   + KO_FD(xx);
            out->ex_re [idx(i,j,v)] =  Iexr  + ADV_FD(exr)  + KO_FD(exr);
            out->ex_im [idx(i,j,v)] =  Iexi  + ADV_FD(exi)  + KO_FD(exi);
            out->bee   [idx(i,j,v)] =  Ibee  + ADV_FD(bee)  + KO_FD(bee);
            out->bxx   [idx(i,j,v)] = -Ibee  + ADV_FD(bxx)  + KO_FD(bxx);
            out->bex_re[idx(i,j,v)] =  Ibexr + ADV_FD(bexr) + KO_FD(bexr);
            out->bex_im[idx(i,j,v)] =  Ibexi + ADV_FD(bexi) + KO_FD(bexi);
#endif
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
        auto k = idx(i,j,v);
        v0->ee    [k] = v1->ee    [k] + a * v2->ee    [k];
        v0->xx    [k] = v1->xx    [k] + a * v2->xx    [k];
        v0->ex_re [k] = v1->ex_re [k] + a * v2->ex_re [k];
        v0->ex_im [k] = v1->ex_im [k] + a * v2->ex_im [k];
        v0->bee   [k] = v1->bee   [k] + a * v2->bee   [k];
        v0->bxx   [k] = v1->bxx   [k] + a * v2->bxx   [k];
        v0->bex_re[k] = v1->bex_re[k] + a * v2->bex_re[k];
        v0->bex_im[k] = v1->bex_im[k] + a * v2->bex_im[k];
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

// v0 = v1 + a * ( v2 + v3 )
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2, const FieldVar * __restrict v3) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
            auto k = idx(i,j,v);
            v0->ee    [k] = v1->ee    [k] + a * (v2->ee    [k] + v3->ee    [k]);
            v0->xx    [k] = v1->xx    [k] + a * (v2->xx    [k] + v3->xx    [k]);
            v0->ex_re [k] = v1->ex_re [k] + a * (v2->ex_re [k] + v3->ex_re [k]);
            v0->ex_im [k] = v1->ex_im [k] + a * (v2->ex_im [k] + v3->ex_im [k]);
            v0->bee   [k] = v1->bee   [k] + a * (v2->bee   [k] + v3->bee   [k]);
            v0->bxx   [k] = v1->bxx   [k] + a * (v2->bxx   [k] + v3->bxx   [k]);
            v0->bex_re[k] = v1->bex_re[k] + a * (v2->bex_re[k] + v3->bex_re[k]);
            v0->bex_im[k] = v1->bex_im[k] + a * (v2->bex_im[k] + v3->bex_im[k]);
        }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::step_rk4() {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

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
#ifdef NVTX
    nvtxRangePop();
#endif
}


