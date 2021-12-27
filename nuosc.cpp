#include "nuosc_class.h"

int main(int argc, char *argv[]) {

    // ====== default input argument ========================
    real dz  = 0.1;
    real dy  = 0.1;
    real y1  = 1,  y0 = -y1;
    real z1  = 4,  z0 = -z1;
    uint nv_in = 40;
    real cfl = 0.4;      real ko = 0.1;

    real mu  = 1.0;
    bool renorm = false;

    real alpha = 0.9;     // nuebar-nue asymmetric parameter
    real lnue  = 0.6;     // width_nue
    real lnueb = 0.53;    // width_nuebar
    real ipt   = 0;       // 0: central_z_perturbation; 1:random
    real eps0  = 1e-2;
    real sigma  = 0.1*z1;    // lzpt = 2*simga**2

    uint ANAL_EVERY = 1; //1.0     / (cfl*dz) + 1;
    uint END_STEP   = 5; //16.0    / (cfl*dz) + 1;
    uint DUMP_EVERY = ANAL_EVERY;

    // ====== Parse input argument ========================
    for (int t = 1; argv[t] != 0; t++) {
        if (strcmp(argv[t], "--dz") == 0 )  {
            dz  = atof(argv[t+1]);     t+=1;
            dy = dz;
        } else if (strcmp(argv[t], "--ymax") == 0 )  {
    	    y1   = atof(argv[t+1]);    t+=1;
            y0   = -y1;
        } else if (strcmp(argv[t], "--zmax") == 0 )  {
            z1   = atof(argv[t+1]);    t+=1;
            z0   = -z1;
        } else if (strcmp(argv[t], "--cfl") == 0 )  {
            cfl   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--nv") == 0 )  {
            nv_in   = atoi(argv[t+1]);    t+=1;
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
            lnue  = atof(argv[t+1]);
            lnueb = atof(argv[t+2]);    t+=2;
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
#ifdef COSENU2D
    int ny  = int((y1-y0)/dy);
#else
    int ny  = 0;
#endif

    // ====== Initialize simuation ========================
#ifdef COSENU2D
    NuOsc state(nv_in, ny, nz, y0, y1, z0, z1, cfl, ko);
    long size=(ny+2*state.gy)*(nz+2*state.gz)*(state.get_nv());
#else
    NuOsc state(nv_in, nz, z0, z1, cfl, ko);
    long size=(nz+2*state.gz)*(state.get_nv());
#endif
    state.set_mu(mu);
    state.set_renorm(renorm);

    state.fillInitValue(ipt, alpha, lnue, lnueb,eps0, sigma);

    // ======  Setup SkimShots ========================
    std::list<real*> plist( { state.P3 } );

    // VZ-Z shot
    //uint skimmed_nv = 4;
    //vector<int> y_slices{ny/2};
    //vector<int> v_slices = gen_skimmed_vslice_index(skimmed_nv, nv_in, state.vy, state.vz );
    //state.addSkimShot(plist, "P3_%06d.bin", DUMP_EVERY, y_slices, nz, v_slices );

    // YZ shot
    uint skimmed_nv = 4;
    uint sy = ny, sz = nz;
    vector<int> vidx = gen_skimmed_vslice_index(skimmed_nv, nv_in, state.vy, state.vz );
    state.addYZSkimShot(plist, "P3XZ_%06d.bin", DUMP_EVERY, sy, sz, vidx);


    // ====== analysis at t=0 ========================
    state.analysis();
    //state.checkSkimShot();
    //state.checkSkimShotToConsole();
    state.checkYZSkimShot();
    
    //state.snapshot();
    //state.write_fz();

    // ====== Begin the main loop ========================
    std::cout << std::flush;
    real stepms;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t=1; t<=END_STEP; t++) {
        //cout << t << "..." << endl;
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            state.analysis();
        }
        if ( t%DUMP_EVERY==0) {
            //state.write_fz();
        }
	//state.checkSkimShot(t);
	//state.checkSkimShotToConsole(t);
        state.checkYZSkimShot(t);

        if (t==10 || t==100 || t==1000 || t==END_STEP) {
	    auto t2 = std::chrono::high_resolution_clock::now();
            stepms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    	    printf("Walltime:  %.3f secs/T, %.3f us per step-grid.\n", stepms/state.phy_time/1000, stepms/t/size*1000);
        }
    }

    printf("Completed.\n");

    printf("[Summ] %d %d %d %d %f\n",  omp_get_max_threads(), ny, nz, state.get_nv(), stepms/END_STEP/size*1000);
    return 0;
}
