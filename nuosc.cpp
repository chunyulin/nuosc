#include "nuosc_class.h"

int main(int argc, char *argv[]) {

#ifdef PAPI
    if ( PAPI_hl_region_begin("computation") != PAPI_OK )
        cout << "PAPI error!" << endl;
#endif

    real dz  = 0.1;
    real dy  = 0.1;
    real y0  = -10;     real y1  =  -y0;
    real z0  = -10;     real z1  =  -z0;
    int nv = 10;
    real cfl = 0.4;      real ko = 0.0;

    real mu  = 0.0;
    bool renorm = false;

    // === initial value
    real alpha = 0.92;     //0.92 for G4b  // nuebar/nue_asymmetric_parameter
    real lnue  = 0.6;     // width_nue
    real lnueb = 0.53;    // width_nuebar
    real ipt   = 0;       // 0_for_central_z_perturbation;1_for_random;2_for_perodic
    real eps0  = 0.1;     // 1e-7 for G4b    // eps0
    real lzpt  = 50.0;    // width_pert_for_0

    int ANAL_EVERY = 5; //10.0   / (cfl*dz) + 1;
    int END_STEP   = 5; //900.0 / (cfl*dz) + 1;
    int DUMP_EVERY = 99999999;

    // Parse input argument
    for (int t = 1; argv[t] != 0; t++) {
        if (strcmp(argv[t], "--dz") == 0 )  {
            dz  = atof(argv[t+1]);     t+=1;
            dy = dz;
        } else if (strcmp(argv[t], "--zmax") == 0 )  {
            z1   = atof(argv[t+1]);    t+=1;
            z0   = -z1;
            y1   = z1;
            y0   = z0;
        } else if (strcmp(argv[t], "--cfl") == 0 )  {
            cfl   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--nv") == 0 )  {
            nv   = atoi(argv[t+1]);    t+=1;
#ifdef CELL_CENTER_V
            assert(nv%2==0);
#else
            assert(nv%2==1);
#endif
        } else if (strcmp(argv[t], "--ko") == 0 )  {
            ko    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--mu") == 0 )  {
            mu    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--renorm") == 0 )  {
            renorm = bool(atoi(argv[t+1]));    t+=1;

            // for monitoring
        } else if (strcmp(argv[t], "--ANA_EVERY") == 0 )  {
            ANAL_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY") == 0 )  {
            DUMP_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--ENDSTEP") == 0 )  {
            END_STEP = atoi(argv[t+1]);    t+=1;
            cout << " ** END_STEP: " << END_STEP << endl;
        } else if (strcmp(argv[t], "--ANA_EVERY_T") == 0 )  {
            ANAL_EVERY = int( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY_T") == 0 )  {
            DUMP_EVERY = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--ENDSTEP_T") == 0 )  {
            END_STEP = int ( atof(argv[t+1]) / (cfl*dz) + 0.5 );    t+=1;
            cout << " ** END_STEP: " << END_STEP << endl;
            // for intial data
        } else if (strcmp(argv[t], "--lnue") == 0 )  {
            lnue  = atof(argv[t+1]);
            lnueb = atof(argv[t+2]);    t+=2;
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
    int ny  = int((y1-y0)/dy);

    // === Initialize simuation
    NuOsc2D state(nv, ny, nz, y0, y1, z0, z1, cfl, ko);
    state.set_mu(mu);
    state.set_renorm(renorm);

    state.fillInitValue(1.0, alpha, lnue, lnueb, ipt, eps0, lzpt);

    // === analysis for t=0
    //state.analysis();
    //state.write_fz();
    //state.write_bin(0);

    std::cout << std::flush;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t=1; t<=END_STEP; t++) {
        cout << t << "..." << endl;
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            state.analysis();
        }
        if ( t%DUMP_EVERY==0) {
            state.write_fz();
            //state.write_bin(t);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now(); 

#ifdef PAPI
    if ( PAPI_hl_region_end("computation") != PAPI_OK )
        cout << "PAPI error!" << endl;
#endif

    printf("Completed.\n");

    long size=(ny+2*state.gy)*(nz+2*state.gz)*(nv);
    real stepms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    printf("Walltime per step          : %.3f ms\n", stepms/END_STEP);
    printf("Core: %d  Walltime per step per grid : %.3f us\n",  omp_get_max_threads(), stepms/END_STEP/size*1000);

    printf("[Summ] %d %d %d %d %f\n",  omp_get_max_threads(), ny, nz, nv, stepms/END_STEP/size*1000);

    return 0;
}
