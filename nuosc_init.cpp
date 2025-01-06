#include "nuosc_class.h"
#include <zlib.h>
// for init data
inline real eps_c(real eps0, real z,   real z0,   real sigma)    { return eps0*std::exp(-(z-z0)*(z-z0)/(2.0*sigma*sigma)); }
inline real eps_r(real eps0, real z=0, real z0=0, real sigma=0 ) { return eps0*rand()/RAND_MAX;}
inline real eps_p(real eps0, real z,   real z0,   real sigma)    { return eps0*(1.0+cos(2*M_PI*(z-z0)/(2.0*sigma*sigma)))*0.5; }

//#define ELN_NUMERICAL_NORMALIZATION
real g(real vx, real vy, real vz, real s[], real v0 = 1.0) {
    #ifdef ELN_NUMERICAL_NORMALIZATION   // default disable
    return std::exp( - (vx-v0)*(vx-v0)/(2.0*s[0]*s[0]) - (vy-v0)*(vy-v0)/(2.0*s[1]*s[1]) - (vz-v0)*(vz-v0)/(2.0*s[2]*s[2]) );
    #else
    // STRANGE: bit different at 14 digits if the factor s[3] is calculated here instead of by passing.
    return std::exp( - (vx-v0)*(vx-v0)/(2.0*s[0]*s[0]) - (vy-v0)*(vy-v0)/(2.0*s[1]*s[1]) - (vz-v0)*(vz-v0)/(2.0*s[2]*s[2]) ) / s[3];
    #endif
}
real g(real v, real sigma, real v0 = 1.0){
    real N = sigma*std::sqrt(0.5*M_PI)*(std::erf((1.0+v0)/sigma/std::sqrt(2.0))+std::erf((1.0-v0)/sigma/std::sqrt(2.0)));
    return std::exp( - (v-v0)*(v-v0)/(2.0*sigma*sigma) ) / N;
}

void NuOsc::restoreInitValue(int restart_from, real alpha, real lnue[], real lnueb[]) {
#ifdef PROFILE
nvtxRangePush(__FUNCTION__);
#endif

  for (int f=0;f<nvar; ++f) {
    // dummy init for OpenMP affinity
    PARFORALL(i,j,k,v) v_stat->wf[f][idx(i,j,k,v)] = 0.0;

    string fname = CKPT+"/it"+std::to_string(restart_from) + "/ckpt" + std::to_string(f) + "." + std::to_string(myrank);
    #ifdef NO_ZLIB
    std::ifstream infile(fname, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
      if (!myrank) cout << "Open file fail! " << fname << endl;
      assert(0);
    }
    infile.read((char *) &iter,     sizeof(uint) );
    infile.read((char *) &phy_time, sizeof(real) );
    #else
    gzFile fp = gzopen(fname.c_str(),"rb");
    if (fp == NULL) {
      if (!myrank) cout << "Open file fail! " << fname << endl;
      assert(0);
    }
    gzread(fp, (char*) &iter,     sizeof(uint));
    gzread(fp, (char*) &phy_time, sizeof(real));
    #endif

    if (iter!=restart_from) assert(0 && "CheckRestart init data fail!");
    if (myrank==0) printf("   Restore from checkpoint %s at iter= %d, time= %f [%d %d %d %d]\n", fname.c_str(), iter, phy_time,nx[0],nx[1],nx[2],nv );

    std::vector<real> carr(nx[0]*nx[1]*nx[2]*nv);
    #ifdef NO_ZLIB
    infile.read(reinterpret_cast<char*>(carr.data()), nx[0]*nx[1]*nx[2]*nv*sizeof(real));
    infile.close();
    #else
    gzread(fp, (char*) carr.data(), nx[0]*nx[1]*nx[2]*nv*sizeof(real));
    gzclose(fp);
    #endif

    PARFORALL(i,j,k,v) {
      v_stat->wf[f][ idx(i,j,k,v) ] = carr[ v + nv*( k + nx[2]*( j + i*nx[1])) ];
    }
  }

  // Recalculate angular distribution ( TODO: consider to read from checkpoint or separate it out )
  real n00=0, n01=0;
  #pragma omp parallel for reduction(+:n00,n01) collapse(3)
  for (int i=0;i<nx[0]; ++i)
  for (int j=0;j<nx[1]; ++j)
  for (int k=0;k<nx[2]; ++k)
  #pragma omp _SIMD_
  for (int v=0;v<nv; ++v) {
    auto ijkv = idx(i,j,k,v);
    // ELN profile
    G0 [ijkv] =         g(vx[v], vy[v], vz[v], lnue );
    G0b[ijkv] = alpha * g(vx[v], vy[v], vz[v], lnueb );
    // initial nv_e
    n00 += vw[v]*v_stat->wf[ff::ee ][ijkv];
    n01 += vw[v]*v_stat->wf[ff::bee][ijkv];
  }

  real n0[] = {n00, n01};
  #ifdef COSENU_MPI
  MPI_Reduce(n0, n_nue0, 2, MPI_MYREAL, MPI_SUM, 0, CartCOMM);
  #else
  n_nue0[0] = n00;
  n_nue0[1] = n01;
  #endif

  n_nue0[0] *= dx*dx*dx*invL;   // initial n_nue
  n_nue0[1] *= dx*dx*dx*invL;   // initial n_nueb

  if (myrank==0) printf("      init number density of nu_e / bnu_e : %g %g\n", n_nue0[0], n_nue0[1]);

#ifdef PROFILE
nvtxRangePop();
#endif
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
        for(int k=-amax;k<=amax;++k) {
            phi[k+amax]=2.0*M_PI*rand()/RAND_MAX;
        }

        #pragma omp parallel for reduction(+:n00,n01) collapse(3)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k){

            real tmpr=0.0, tmpi=0.0;
            for(int q=-amax;q<amax;++q) {
                if(q!=0){
                    tmpr += 1.e-7/abs(q)*cos(pi2oL*q*X[DIM-1][k] + phi[q+amax]);
                    tmpi += 1.e-7/abs(q)*sin(pi2oL*q*X[DIM-1][k] + phi[q+amax]);
                }
            }

            real p3o=sqrt(1.0-tmpr*tmpr-tmpi*tmpi);
            for (int v=0;v<nv; ++v){
                auto kv = idx(i,j,k,v);

                // ELN profile
                G0 [kv] =         g(vx[v], vy[v], vz[v], lnue );
                G0b[kv] = alpha * g(vx[v], vy[v], vz[v], lnueb);
                v_stat->wf[ff::ee]  [kv] =  0.5* G0 [kv]*(1.0+p3o);//sqrt(f0*f0 - (v_stat->ex_re[idx(i,j)])*(v_stat->ex_re[idx(i,j)]));
                v_stat->wf[ff::mm]  [kv] =  0.5* G0 [kv]*(1.0-p3o);
                v_stat->wf[ff::emr] [kv] =  0.5* G0 [kv]*tmpr;//1e-6;
                v_stat->wf[ff::emi] [kv] =  0.5* G0 [kv]*tmpi;//random_amp(0.001);
                v_stat->wf[ff::bee] [kv] =  0.5* G0b[kv]*(1.0+p3o);
                v_stat->wf[ff::bmm] [kv] =  0.5* G0b[kv]*(1.0-p3o);
                v_stat->wf[ff::bemr][kv] =  0.5* G0b[kv]*tmpr;//1e-6;
                v_stat->wf[ff::bemi][kv] = -0.5* G0b[kv]*tmpi;//random_amp(0.001);
                // initial nv_e
                n00 += vw[v]*v_stat->wf[ff::ee] [kv];
                n01 += vw[v]*v_stat->wf[ff::bee][kv];
            }
        }

    } else {  // init data homogeneous in DIM=ipt.

        Vec ng(nv), ngb(nv);

#ifdef ELN_NUMERICAL_NORMALIZATION
        real ing0=0, ing1=0;
        #pragma omp parallel for simd reduction(+:ing0, ing1)
        for (int v=0;v<nv;++v) {
            ng [v] = g(vx[v], vy[v], vz[v], lnue );
            ngb[v] = g(vx[v], vy[v], vz[v], lnueb );
            ing0 += vw[v]*ng [v];
            ing1 += vw[v]*ngb[v];
        }
        cout << "Normalization: "<< std::setprecision(16) << ing0 << " " << ing1 << endl;
        #pragma omp parallel for simd
        for (int v=0;v<nv;++v) {
            ng [v] /= ing0;
            ngb[v] /= ing1;
        }
#else
        #pragma omp parallel for simd
        for (int v=0;v<nv;++v) {
            ng [v] = g(vx[v], vy[v], vz[v], lnue );
            ngb[v] = g(vx[v], vy[v], vz[v], lnueb );
        }
#endif

        real (*spatialeps)(real,real,real,real);
        if      (ipt<4) {          // center Z perturbation
            if (myrank==0) printf("   Init data: [%s] alpha= %f eps= %g sigma= %g lnu:[ %g %g %g ]  lnub:[ %g %g %g ]\n", "Point-like pertur", alpha, eps0, sigma, lnue[0],lnue[1],lnue[2], lnueb[0],lnueb[1],lnueb[2] );
            spatialeps = &eps_c; 
        } else if (ipt==5) {       // random
            if (myrank==0) printf("   Init data: [%s] alpha= %f eps= %g sigma= %g lnu:[ %g %g %g ]  lnub:[ %g %g %g ]\n", "Random pertur", alpha, eps0, sigma, lnue[0],lnue[1],lnue[2], lnueb[0],lnueb[1],lnueb[2] );
            spatialeps = &eps_r; 
        } else if (ipt==6) {       // periodic Z perturbation
            if (myrank==0) printf("   Init data: [%s] alpha= %f eps= %g sigma= %g lnu:[ %g %g %g ]  lnub:[ %g %g %g ]\n", "Periodic Z", alpha, eps0, sigma, lnue[0],lnue[1],lnue[2], lnueb[0],lnueb[1],lnueb[2] );
            spatialeps = &eps_p;
        } else             { assert(0); }   // Not implemented

        #pragma omp parallel for reduction(+:n00,n01) collapse(3)
        #pragma acc parallel loop reduction(+:n00,n01) collapse(3)
        for (int i=0;i<nx[0]; ++i)
        for (int j=0;j<nx[1]; ++j)
        for (int k=0;k<nx[2]; ++k)
        #pragma omp _SIMD_
        #pragma acc for
        for (int v=0;v<nv; ++v) {
            auto ijkv = idx(i,j,k,v);

            // ELN profile
            G0 [ijkv] =         ng [v];
            G0b[ijkv] = alpha * ngb[v];

            real tmpr = spatialeps(eps0, X[ipt][k], 0., sigma);
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

#ifdef COSENU_MPI
        real n0[] = {n00, n01};
        MPI_Reduce(n0, n_nue0, 2, MPI_MYREAL, MPI_SUM, 0, CartCOMM);
#else
        n_nue0[0] = n00;
        n_nue0[1] = n01;
#endif

#if 0
    // dumpG
    std::ofstream o;
    char fn[32];
    sprintf(fn, "G0_%f.dat", alpha);
    o.open(fn, std::ofstream::out | std::ofstream::trunc);

    o << "## nv" << endl;
    for (int v=0;v<nv;++v) o << vw[v] << " ";
    o << endl;

    o << "## ng/ngb" << endl;
    for (int v=0;v<nv;++v) o << ng [v] << " " << ngb[v] << " ";
    o << endl;

    o << "## G0/G0b" << endl;
    for (int v=0;v<nv;v++) {
      auto ijv = idx(1,1,1,v);
      o << vz[v] << " " << std::setprecision(15) <<  G0[ijv] << " " << G0b[ijv] << endl;
    }
    o.close();
#endif

    } //  end select case (ipt)

    n_nue0[0] *= dx*dx*dx*invL;   // initial n_nue
    n_nue0[1] *= dx*dx*dx*invL;   // initial n_nueb

    if (myrank==0) printf("      init number density of nu_e: %g %g\n", n_nue0[0], n_nue0[1]);

    //Dump2Text("dump.dat");

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
