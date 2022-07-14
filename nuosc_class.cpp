#include "nuosc_class.h"

void NuOsc::updatePeriodicBoundary(FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    // Assume cell-center:     [-i=nz-i,-1=nz-1] ,0,...,nz-1, [nz=0, nz+i=i]

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
    for (int i=0;i<1; i++)
    for (int j=0;j<0; j++)
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

#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::updateInjetOpenBoundary(FieldVar * __restrict in) { 
}


void NuOsc::calRHS(FieldVar * __restrict out, const FieldVar * __restrict in) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    auto rx = 2/dz;   // J=dz/2= (dx/dr) where x:phy-point r:LGL point

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(2) gang
    for (int i=0;i<nz; ++i)
    for (int p=0;p<np; ++p) {
#pragma acc loop vector
        for (int v=0;v<nv; ++v) {
            
            long ipv = idx(i,p,v);
            
            // f* = {vu} + n.|v|*(1-alpha)*[u]/2
            // RHS[p] = VVT[p][1] * (f[np-1]-f*[np-1]) - VVT[p][0] * (f[0]-f*[0])
            //        = VVT[p][1] * (v*u[Rm]-f*[np-1]) - VVT[p][0] * (v*u[0]-f*[0])
            //        = - 0.5* ( vz[v]+std::abs(vz[v])*(1-flux_alpha) )* ( VVT[p][1]*(u[Rp]-u[Rm]) - VVT[p][0]*(u[Lp]-u[Lm]))
        
            // indeies including periodic boundary
            uint Rplus =  idx( (i+1)%nz,      0, v);
            uint Rminus = idx( i,          np-1, v);
            uint Lplus =  idx( i,             0, v);
            uint Lminus = idx((nz+i-1)%nz, np-1, v);

            // Boundary terms
            real fa = 0.5*std::abs(vz[v])*(1-flux_alpha);
            real fR = - 0.5* (   vz[v] - fa ) * VVT[p][1];   // OK
            real fL = - 0.5* ( - vz[v] - fa ) * VVT[p][0];   // OK
            real Bee    = fR*(in->ee    [Rplus]-in->ee    [Rminus]) - fL*(in->ee    [Lplus]-in->ee    [Lminus]);
            real Bxx    = fR*(in->xx    [Rplus]-in->xx    [Rminus]) - fL*(in->xx    [Lplus]-in->xx    [Lminus]);
            real Bexr   = fR*(in->ex_re [Rplus]-in->ex_re [Rminus]) - fL*(in->ex_re [Lplus]-in->ex_re [Lminus]);
	    real Bexi   = fR*(in->ex_im [Rplus]-in->ex_im [Rminus]) - fL*(in->ex_im [Lplus]-in->ex_im [Lminus]);
            real Bbee   = fR*(in->bee   [Rplus]-in->bee   [Rminus]) - fL*(in->bee   [Lplus]-in->bee   [Lminus]);
            real Bbxx   = fR*(in->bxx   [Rplus]-in->bxx   [Rminus]) - fL*(in->bxx   [Lplus]-in->bxx   [Lminus]);
	    real Bbexr  = fR*(in->bex_re[Rplus]-in->bex_re[Rminus]) - fL*(in->bex_re[Lplus]-in->bex_re[Lminus]);
	    real Bbexi  = fR*(in->bex_im[Rplus]-in->bex_im[Rminus]) - fL*(in->bex_im[Lplus]-in->bex_im[Lminus]);

            // Advection terms:  RHS[p]  = vz[v]*D[p][q]*u[q,v]
            real Dee   = 0;
            real Dxx   = 0;
            real Dexr  = 0;
            real Dexi  = 0;
            real Dbee  = 0;
            real Dbxx  = 0;
            real Dbexr = 0;
            real Dbexi = 0;
#if 1
	    for (int q=0;q<np; ++q) {
    	        auto iqv = idx(i,q,v);
    	        Dee   += Dr[p][q]*(in->ee    [iqv]);
    	        Dxx   += Dr[p][q]*(in->xx    [iqv]);
    	        Dexr  += Dr[p][q]*(in->ex_re [iqv]);
    	        Dexi  += Dr[p][q]*(in->ex_im [iqv]);
    	        Dbee  += Dr[p][q]*(in->bee   [iqv]);
    	        Dbxx  += Dr[p][q]*(in->bxx   [iqv]);
    	        Dbexr += Dr[p][q]*(in->bex_re[iqv]);
    	        Dbexi += Dr[p][q]*(in->bex_im[iqv]);
    	    }
#endif
            // prepare v-integral with a cubature rule
            real Iee    = 0;
            real Iexr   = 0;
            real Iexi   = 0;
            real Ibee   = 0;
            real Ibexr  = 0;
            real Ibexi  = 0;

            // The base pointer for this stencil
            real xx_m_ee   = in->xx [ipv] - in->ee [ipv];
            real bxx_m_bee = in->bxx[ipv] - in->bee[ipv];
            real exr   = in->ex_re [ipv];
            real exi   = in->ex_im [ipv];
            real bexr  = in->bex_re[ipv];
            real bexi  = in->bex_im[ipv];

            #pragma acc loop
            for (int k=0;k<nv; k++) {   // vz' integral
    	        auto ipk = idx(i,p,k);
                real eep_m_xxp_m_beep_p_bxxp = (in->ee[ipk]) - (in->xx[ipk]) - (in->bee[ipk]) + (in->bxx[ipk]);
                real expr_m_bexpr = in->ex_re [ipk] - in->bex_re[ipk];
                real expi_p_bexpi = in->ex_im [ipk] + in->bex_im[ipk];

                // terms for -i* mu * [rho'-rho_bar', rho]
                real fvdv = vw[k]*mu* (1-vz[v]*vz[k]);
                Iee   +=  2* fvdv * (       exr * expi_p_bexpi -  exi * expr_m_bexpr );
                Iexr  +=     fvdv * (   xx_m_ee * expi_p_bexpi +  exi * eep_m_xxp_m_beep_p_bxxp );
                Iexi  +=     fvdv * ( - xx_m_ee * expr_m_bexpr -  exr * eep_m_xxp_m_beep_p_bxxp );
                Ibee  +=  2* fvdv * (      bexr * expi_p_bexpi + bexi * expr_m_bexpr );
                Ibexr +=     fvdv * ( bxx_m_bee * expi_p_bexpi - bexi * eep_m_xxp_m_beep_p_bxxp );
                Ibexi +=     fvdv * ( bxx_m_bee * expr_m_bexpr + bexr * eep_m_xxp_m_beep_p_bxxp );
            }

#ifdef VACUUM_OFF
            // All RHS with terms for -i [H0, rho], advector, v-integral, etc...
            out->ee    [ipv] = (Bee   + Iee   - vz[v]*Dee   ) * rx;
            out->xx    [ipv] = (Bee   - Iee   - vz[v]*Dxx   ) * rx;
            out->ex_re [ipv] = (Bexr  + Iexr  - vz[v]*Dexr  ) * rx;
            out->ex_im [ipv] = (Bexi  + Iexi  - vz[v]*Dexi  ) * rx;
            out->bee   [ipv] = (Bbee  + Ibee  - vz[v]*Dbee  ) * rx;
            out->bxx   [ipv] = (Bbee  - Ibee  - vz[v]*Dbxx  ) * rx;
            out->bex_re[ipv] = (Bbexr + Ibexr - vz[v]*Dbexr ) * rx;
            out->bex_im[ipv] = (Bbexi + Ibexi - vz[v]*Dbexi ) * rx;
#else
            out->ee    [ipv] = (Bee   + Iee   - vz[v]*Dee   - pmo* 2*st*exi ) * rx;
            out->xx    [ipv] = (Bee   - Iee   - vz[v]*Dxx   + pmo* 2*st*exi ) * rx;
            out->ex_re [ipv] = (Bexr  + Iexr  - vz[v]*Dexr  - pmo* 2*ct*exi ) * rx;
            out->ex_im [ipv] = (Bexi  + Iexi  - vz[v]*Dexi  + pmo*(2*ct*exr  - st* xx_m_ee )) * rx;
            out->bee   [ipv] = (Bbee  + Ibee  - vz[v]*Dbee  - pmo* 2*st*bexi);
            out->bxx   [ipv] = (Bbee  - Ibee  - vz[v]*Dbxx  + pmo* 2*st*bexi);
            out->bex_re[ipv] = (Bbexr + Ibexr - vz[v]*Dbexr - pmo* 2*ct*bexi);
            out->bex_im[ipv] = (Bbexi + Ibexi - vz[v]*Dbexi + pmo*(2*ct*bexr - st* bxx_m_bee )) * rx;
#endif
        } // end for v
    }  // end for i,p
    
#if 0
    cout << "================= " << vz[nv-1] << endl;
    for (int i=0;i<nz; ++i) {
    for (int p=0;p<np; ++p) {
        cout << out->ee[idx(i,p,nv-1)] << "  ";
    }
    cout << endl;
    }    
#endif
    
#ifdef NVTX
    nvtxRangePop();
#endif
}

/* v0 = v1 + a * v2 */
void NuOsc::vectorize(FieldVar* __restrict v0, const FieldVar * __restrict v1, const real a, const FieldVar * __restrict v2) {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    PARFORALL(i,p,v) {
        auto k = idx(i,p,v);
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

    PARFORALL(i,p,v) {
            auto k = idx(i,p,v);
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


