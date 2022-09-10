#include "nuosc_class.h"
#include "utils.h"
#include <limits>

int main(int argc, char *argv[]) {

    real dz  =  0.1;
    real dx  =  0.1;
    real x0  = -0.05;     real x1  =  -x0;
    real z0  = - 10;      real z1  =  -z0;
    int nv_in = 33, nphi = 32;
    real cfl = 0.4;      real ko = 0.0;

    real mu  = 1.0;
    real pmo = 1.0;
    bool renorm = false;

    // === initial value
    real alpha = 0.9;                   // nuebar-nue asymmetric parameter
    real lnue  = 0.6,   lnueb  = 0.53;  // width of nu/nubar in z
    real lnuex  = std::numeric_limits<real>::max();
    real lnuebx = std::numeric_limits<real>::max();
    real ipt   = 0;                     // 0: central_z_perturbation; 1:random; 4:noc case
    real eps0  = 0.1;
    real sigma  = 100.0;    // lzpt = 2*simga**2

    int ANAL_EVERY = 5;    // 10.0  / (cfl*dz) + 1;
    int END_STEP   = 5;    // 900.0 / (cfl*dz) + 1;
    int DUMP_EVERY = 99999999;

    // Parse input argument --------------------------------------------
    for (int t = 1; argv[t] != 0; t++) {
        if (strcmp(argv[t], "--dz") == 0 )  {
            dz  = atof(argv[t+1]);     t+=1;
        } else if (strcmp(argv[t], "--dx") == 0 )  {
            dx  = atof(argv[t+1]);     t+=1;
        } else if (strcmp(argv[t], "--xmax") == 0 )  {
            x1   = atof(argv[t+1]);    t+=1;
            x0   = -x1;
        } else if (strcmp(argv[t], "--zmax") == 0 )  {
            z1   = atof(argv[t+1]);    t+=1;
            z0   = -z1;
        } else if (strcmp(argv[t], "--cfl") == 0 )  {
            cfl   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--nv") == 0 )  {
            nv_in   = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--nphi") == 0 )  {
            nphi    = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ko") == 0 )  {
            ko    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--mu") == 0 )  {
            mu    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--pmo") == 0 )  {
            pmo   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--renorm") == 0 )  {
            renorm = bool(atoi(argv[t+1]));    t+=1;

            // for monitoring
        } else if (strcmp(argv[t], "--ANA_EVERY") == 0 )  {
            ANAL_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY") == 0 )  {
            DUMP_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--END_STEP") == 0 )  {
            END_STEP = atoi(argv[t+1]);    t+=1;
            cout << " ** END_STEP: " << END_STEP << endl;
        } else if (strcmp(argv[t], "--ANA_EVERY_T") == 0 )  {
            ANAL_EVERY = int( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY_T") == 0 )  {
            DUMP_EVERY = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--END_STEP_T") == 0 )  {
            END_STEP = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** END_STEP: " << END_STEP << endl;
            // for intial data
        } else if (strcmp(argv[t], "--lnue") == 0 )  {
            lnue   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnueb") == 0 )  {
            lnueb  = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnuex") == 0 )  {
            lnuex   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnuebx") == 0 )  {
            lnuebx  = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--sigma") == 0 )  {
            sigma = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--eps0") == 0 )  {
            eps0 = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--alpha") == 0 )  {
            alpha = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ipt") == 0 )  {
            ipt = atoi(argv[t+1]);    t+=1;
        } else {
            printf("Unreconganized parameters %s!\n", argv[t]);
            exit(0);
        }
    }
    int nz  = int((z1-z0)/dz);
    int nx  = int((x1-x0)/dx);

    // === Initialize simuation
#ifdef COSENU2D
    NuOsc state(nv_in, nphi, nx, nz, x0, x1, z0, z1, cfl, ko);
    long size=(nx+2*state.gx)*(nz+2*state.gz)*(state.get_nv());
#else
    NuOsc state(nv_in, nz, z0, z1, cfl, ko);
    long size=(nz+2*state.gz)*(state.get_nv());
#endif
    state.set_mu(mu);
    state.set_pmo(pmo);
    state.set_renorm(renorm);

#ifdef ADV_TEST
    if      (ipt==10) state.fillInitGaussian( eps0, sigma);
    else if (ipt==20) state.fillInitSquare( eps0, sigma);
    else if (ipt==30) state.fillInitTriangle( eps0, sigma);
#else
    state.fillInitValue(ipt, alpha, eps0, sigma, lnue, lnueb, lnuex, lnuebx);
#endif

    // === analysis for t=0
    state.analysis();

    // ======  Setup 1D output  ========================
    if (DUMP_EVERY <= END_STEP) {
        std::list<real*> vlist( { state.P3 } );
        std::vector<int> vslice;
        for (int v=0;v<nv_in;++v) {
            vslice.push_back( int((nv_in-1)/2)*nv_in + v );
        }
        //state.addSnapShotAtV(vlist, "P3_%06d.bin", DUMP_EVERY, vslize );
        state.addSnapShotAtXV(vlist, "P3_%06d.bin", DUMP_EVERY, std::vector<int>{0,nx/2,nx-1}, vslice );
        //std::list<real*> plist( { state.P3 } );
        //state.addSkimShot(plist, "P3_%06d.bin", DUMP_EVERY, nz, 11 );
        //std::list<real*> rlist( {state.v_stat->ee, state.v_stat->xx} );
        //state.addSkimShot(rlist, "Rho%06d.bin", DUMP_EVERY, 10240, 21 );

#ifdef ADV_TEST
        std::list<real*> vlist( { state.v_stat->ee } );
        state.addSnapShotAtV(vlist, "ee%06d.bin", DUMP_EVERY,  std::vector<int>{0,state.get_nv()/2, state.get_nv()-1} );
        //state.addSnapShotAtV(vlist, "ee%06d.bin", DUMP_EVERY, gen_skimmed_vslice_index(nv_in, nv_in)  );
#endif
        state.checkSnapShot(0);
        //state.checkSkimShots();
        //state.snapshot();
        //state.write_fz();
    }    

    std::cout << std::flush;
    real stepms;
    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    const int cooltime = 5;
    for (int t=1; t<=END_STEP; t++) {

        if (t==cooltime)  t1 = std::chrono::high_resolution_clock::now();
        //cout << t << "..." << endl;
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            state.analysis();
        }

        state.checkSnapShot(t);

        if ( t==10 || t==100 || t==1000 || t==END_STEP) {
            auto t2 = std::chrono::high_resolution_clock::now();
            stepms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    	    printf("%d Walltime:  %.3f secs/T, %.2f ns per step-grid.\n", t, stepms/state.phy_time/1000, stepms/(t-cooltime+1)/size*1e6);
        }
    }

    printf("Completed.\n");
    printf("Memory usage (MB): %.2f\n", getMemoryUsage()/1024.0);

    printf("[Summ] %d %d %d %d %f\n",  omp_get_max_threads(), nx, nz, state.get_nv(), stepms/(END_STEP-cooltime+1)/size*1e6);
    return 0;
}
