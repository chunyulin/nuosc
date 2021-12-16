#include "nuosc_class.h"

// for init data
inline real eps_c(real z, real z0, real eps0, real lzpt){return eps0*exp(-(z-z0)*(z-z0)/lzpt);}
inline real eps_r(real z, real z0, real eps0) {return eps0*rand()/RAND_MAX;}
double g(double v, double v0, double sigma){
    double exponant = (v-v0)*(v-v0)/(2.0*sigma*sigma);
    double N = sigma*sqrt(M_PI/2.0)*(erf((1.0+v0)/sigma/sqrt(2.0))+erf((1.0-v0)/sigma/sqrt(2.0)));
    return exp(-exponant)/N;
}

int gen_v_simple(const int nv, real *& vw, real *& vy, real *& vz) {
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


void NuOsc2D::fillInitValue(real f0, real alpha, real lnue, real lnueb, int ipt, real eps0, real lzpt) {

    printf("   Init data: eps = %g  alpha = %f\n", eps0, alpha);

    FORALL(i,j,v) {
            G0 [idx(i,j,v)] =         g(vz[j], 1.0, lnue )*0.5;
            G0b[idx(i,j,v)] = alpha * g(vz[j], 1.0, lnueb)*0.5;
    }

    FORALL(i,j,v) {
            real tmp;
            if(ipt==0) { tmp = eps_c(Z[j],0.0,eps0,lzpt); }
            if(ipt==1) { tmp = eps_r(Z[j],0.0,eps0); }
            real p3o = sqrt(1.0-tmp*tmp);
            v_stat->ee    [idx(i,j,v)] = G0[idx(i,j,v)]*(1.0+p3o); //sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
            v_stat->xx    [idx(i,j,v)] = G0[idx(i,j,v)]*(1.0-p3o);
            v_stat->ex_re [idx(i,j,v)] = G0[idx(i,j,v)]*tmp;
            v_stat->ex_im [idx(i,j,v)] = 0.0; //random_amp(0.001);
            v_stat->bee   [idx(i,j,v)] = G0b[idx(i,j,v)]*(1.0+p3o);
            v_stat->bxx   [idx(i,j,v)] = G0b[idx(i,j,v)]*(1.0-p3o);
            v_stat->bex_re[idx(i,j,v)] = G0b[idx(i,j,v)]*tmp;
            v_stat->bex_im[idx(i,j,v)] = 0.0; //random_amp(0.001);
    }

#ifdef BC_PERI
    updatePeriodicBoundary(v_stat);
#else
    updateInjetOpenBoundary(v_stat);
#endif

}


void NuOsc2D::write_bin(const int t) {
    char filename[32];
    sprintf(filename,"stat_%04d.bin", t);

    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) {
        cout << "*** Open fails: " << filename << endl;
    }

    int size = (ny+2*gy)*(nz+2*gz)*nv;

    FieldVar *v = v_stat; 

    outfile.write((char *) &phy_time,  sizeof(real));
    outfile.write((char *) &vz[0],     sizeof(real));
    outfile.write((char *) &vz[nv-1], sizeof(real));
    outfile.write((char *) &Z[0],      sizeof(real));
    outfile.write((char *) &Z[nz-1],   sizeof(real));
    outfile.write((char *) &nz,  sizeof(int));
    outfile.write((char *) &nv, sizeof(int));
    outfile.write((char *) &gz,  sizeof(int));
    outfile.write((char *) v->ee,     size*sizeof(real));
    outfile.write((char *) v->xx,     size*sizeof(real));
    outfile.write((char *) v->ex_re,  size*sizeof(real));
    outfile.write((char *) v->ex_im,  size*sizeof(real));
    outfile.write((char *) v->bee,    size*sizeof(real));
    outfile.write((char *) v->bxx,    size*sizeof(real));
    outfile.write((char *) v->bex_re, size*sizeof(real));
    outfile.write((char *) v->bex_im, size*sizeof(real));
    outfile.close();
    printf("		Write %d x %d into %s\n", nv, nz+2*gz, filename);
}


void NuOsc2D::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for collapse(3)
#pragma acc parallel for collapse(3)
    for (int i=0;i<ny; i++)
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

#pragma omp parallel for collapse(3)
#pragma acc parallel for collapse(3)
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

#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc2D::updateInjetOpenBoundary(FieldVar * __restrict in) { 
    cout << "Not implemented." << endl;
    assert(0);
}


void NuOsc2D::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    const int nzv = nv*nz;

#pragma omp parallel for collapse(3)
#pragma acc parallel loop gang
    for (int i=0;i<ny; i++)
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
            const real ko_eps_y = -ko/dy/64.0;
            const real ko_eps_z = -ko/dz/64.0;
            #define KO_FD2D(x) ( ko_eps_z*( x[-3*nv]  + x[3*nv]  - 6*(x[-2*nv]+x[2*nv])   + 15*(x[-nv]+x[nv])   - 20*x[0] ) + \
                                 ko_eps_y*( x[-3*nzv] + x[3*nzv] - 6*(x[-2*nzv]+x[2*nzv]) + 15*(x[-nzv]+x[nzv]) - 20*x[0] )  )
#else
            // Kreiss-Oliger dissipation (3-nd order)
            real ko_eps_y = -ko/dy/16.0;
            real ko_eps_z = -ko/dz/16.0;
	    #define KO_FD2D(x) (  ko_eps_z * ( x[-2*nv]  + x[2*nv]  - 4*(x[-nv] +x[nv])  + 6*x[0] ) + \
                                  ko_eps_y * ( x[-2*nzv] + x[2*nzv] - 4*(x[-nzv]+x[nzv]) + 6*x[0] )  )
#endif

            // prepare advection FD operator
#if defined(ADVEC_CENTER_FD)
            //   4-th order FD for 1st-derivation ~~ ( (a[-2]-a[2])/12 - 2/3*( a[-1]-a[1]) ) / dx
            real factor_y = -vy[i]/(12*dy);
            real factor_z = -vz[i]/(12*dz);
            #define ADV_FD2D(x)  factor_z*(  (x[-2*nv] -x[2*nv])  - 8.0*(x[-nv] -x[nv])  ) + \
                                 factor_y*(  (x[-2*nzv]-x[2*nzv]) - 8.0*(x[-nzv]-x[nzv] ) )
#elif defined(ADVEC_UPWIND)
            int sv = sgn(vz[i]);
            real factor = -sv*vz[i]/(12*dz);
            #define ADV_FD(x)  ( -4*x[-3*sv*nv] + 18*x[-2*sv*nv] - 36*x[-sv*nv] + 22*x[0] )
#else
            // advection term: (4-th order lopsided finite differencing)
            int sv = sgn(vz[i]);
            real factor = sv*vz[i]/(12*dz);
            #define ADV_FD(x)  ( x[-3*sv*nv] - 6*x[-2*sv*nv] + 18*x[-sv*nv] - 10*x[0] - 3*x[sv*nv] )
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
                Iee   +=  2*vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  (        exr[0] *(expi + bexpi) -  exi[0]*(expr- bexpr) );
                Iexr  +=    vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  (  (xx[0]-ee[0])*(expi + bexpi) +  exi[0]*(eep - xxp - beep + bxxp) );
                Iexi  +=    vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  ( -(xx[0]-ee[0])*(expr - bexpr) -  exr[0]*(eep - xxp - beep + bxxp) );
                Ibee  +=  2*vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  (       bexr[0] *(expi + bexpi) + bexi[0]*(expr- bexpr) );
                Ibexr +=    vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expi + bexpi) - bexi[0]*(eep - xxp - beep + bxxp) );
                Ibexi +=    vw[k]*mu* (1-vy[i]*vy[k]-vz[i]*vz[k])*  ((bxx[0]-bee[0])*(expr - bexpr) + bexr[0]*(eep - xxp - beep + bxxp) );
            }

            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [idx(i,j,v)] =  Iee   + ADV_FD2D(ee)   + KO_FD2D(ee)   - pmo* 2*st*exi [0];
            out->xx    [idx(i,j,v)] = -Iee   + ADV_FD2D(xx)   + KO_FD2D(xx)   + pmo* 2*st*exi [0];
            out->ex_re [idx(i,j,v)] =  Iexr  + ADV_FD2D(exr)  + KO_FD2D(exr)  - pmo* 2*ct*exi [0];
            out->ex_im [idx(i,j,v)] =  Iexi  + ADV_FD2D(exi)  + KO_FD2D(exi)  + pmo*(2*ct*exr [0] + st*( ee[0] - xx[0] ) );
            out->bee   [idx(i,j,v)] =  Ibee  + ADV_FD2D(bee)  + KO_FD2D(bee)  - pmo* 2*st*bexi[0];
            out->bxx   [idx(i,j,v)] = -Ibee  + ADV_FD2D(bxx)  + KO_FD2D(bxx)  + pmo* 2*st*bexi[0];
            out->bex_re[idx(i,j,v)] =  Ibexr + ADV_FD2D(bexr) + KO_FD2D(bexr) - pmo* 2*ct*bexi[0];
            out->bex_im[idx(i,j,v)] =  Ibexi + ADV_FD2D(bexi) + KO_FD2D(bexi) + pmo*(2*ct*bexr[0] + st*( bee[0] - bxx[0] ) );

        }
#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc2D::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
        int k = idx(i,j,v);
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
void NuOsc2D::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2, const FieldVar * __restrict v3) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,j,v) {
            int k = idx(i,j,v);
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


void NuOsc2D::step_rk4() {
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


