#include "nuosc_class.h"
#include "utils.h"

inline bool ifclose(real t, real dt, real step) {
    return ( step > 0 && std::abs(t - std::round(t/step)*step) <= 0.5*dt );
}

int main(int argc, char *argv[]) {

    int ord = 5;

    // ====== default input argument ========================
    real dz  = 0.5;       // I prefer fixing only dz and [z0, z1], with nz the direvatives. 
    real dy  = 0.5;
    real y1  = 4,  y0 = -y1;
    real z1  = 4,  z0 = -z1;
#ifdef COSENU2D
    uint nv_in = 5;     // For 2D, the actuall nv may not be nv_in^2...
#else
    uint nv_in = 5;     // For 2D, the actuall nv may not be nv_in^2...
#endif
    real cfl = 0.75;
    real mu  = 1.0;
    bool renorm = false;

    real alpha = 0.9;     // nuebar-nue asymmetric parameter
    real lnue  = 0.6;     // width_nue
    real lnueb = 0.53;    // width_nuebar
    real ipt   = 0;       // 0: central_z_perturbation; 1:random
    real eps0  = 1e-3;  //0.1*z1;
    real sigma  = 5; ////0.1*z1;    // lzpt = 2*simga**2

    uint ANA_EVERY = 0.2       / (cfl*dz) + 1;
    uint END_STEP   = 0.2*z1    / (cfl*dz) + 1;
    uint DUMP_EVERY = ANA_EVERY;

    uint sy = 1; // output dimension of skimmed shot
    uint sz = 1;
    uint sv = 1;

    real ANA_EVERY_T = -1;
    real DUMP_EVERY_T = -1;
    real END_STEP_T = -1;

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
        } else if (strcmp(argv[t], "--mu") == 0 )  {
            mu    = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--renorm") == 0 )  {
            renorm = bool(atoi(argv[t+1]));    t+=1;
        
        } else if (strcmp(argv[t], "--ord") == 0 )  {
            ord = atoi(argv[t+1]);    t+=1;

        // for monitoring
        } else if (strcmp(argv[t], "--ANA_EVERY") == 0 )  {
            ANA_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** ANA_EVERY: " << ANA_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY") == 0 )  {
            DUMP_EVERY = atoi(argv[t+1]);    t+=1;
            cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--END_STEP") == 0 )  {
            END_STEP = atoi(argv[t+1]);    t+=1;
            cout << " ** END_STEP: " << END_STEP << endl;
        } else if (strcmp(argv[t], "--ANA_EVERY_T") == 0 )  {
            ANA_EVERY_T = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--DUMP_EVERY_T") == 0 )  {
            DUMP_EVERY_T = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--END_STEP_T") == 0 )  {
            END_STEP_T = atof(argv[t+1]);    t+=1;

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

	// for skimmed shot
        } else if (strcmp(argv[t], "--sy") == 0 )  {
            sy = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--sz") == 0 )  {
            sz = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--sv") == 0 )  {
            sv = atoi(argv[t+1]);    t+=1;

        } else {
            printf("Unreconganized parameters %s!\n", argv[t]);
            exit(0);
        }
    }
    int nz  = int((z1-z0)/dz);
    int ny  = 1;
    // ====== Initialize simuation ========================
    NuOsc state(ord, nv_in, nz, z0, z1, cfl);

    if (END_STEP_T > 0) {
        END_STEP = int( END_STEP_T / state.get_dt() + 0.5 );
        cout << " ** END_STEP: " << END_STEP << endl;
    }

    long size=nz*(state.get_np())*(state.get_nv());
    state.set_mu(mu);
    state.set_renorm(renorm);
#ifdef ADV_TEST
    if (ipt==10) state.fillInitGaussian( eps0, sigma);
    if (ipt==20) state.fillInitSquare( eps0, sigma);
    if (ipt==30) state.fillInitTriangle( eps0, sigma);
#else
    state.fillInitValue(ipt, alpha, lnue, lnueb,eps0, sigma);
#endif
    printf("Memory usage (MB): %.2f\n", getMemoryUsage()/1024.0);

    // ======  Setup 1D output  ========================
    std::list<real*> vlist( { state.v_stat->ee.data(), state.v_stat->ex_re.data() , state.v_stat->ex_im.data() } );
    state.addSnapShotAtV(vlist, "ee%06d.bin", DUMP_EVERY, std::vector<int>{state.get_nv()-1} );
    state.takeSnapShot(0);
                                
    // ====== analysis at t=0 ========================
    state.analysis();
    
    // ====== Begin the main loop ========================
    std::cout << std::flush;
    real stepms;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int t=1; t<=END_STEP; t++) {
        //cout << t << "..." << endl;
        state.step_rk4();

        //if ( t%ANA_EVERY==0)  {
        if (ifclose(state.phy_time, state.get_dt(), ANA_EVERY_T)) {
            state.analysis();
        }

        //if ( t%DUMP_EVERY==0) {
        if (ifclose(state.phy_time, state.get_dt(), DUMP_EVERY_T)) {
            //state.write_fz();
            state.takeSnapShot(t);
        }

        if (t==1 || t==10 || t==100 || t==1000 || t==END_STEP) {
	    auto t2 = std::chrono::high_resolution_clock::now();
            stepms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    	    printf("Walltime:  %.3f secs/T, %.3f us per step-grid.\n", stepms/state.phy_time/1000, stepms/t/size*1000);
        }
    }

    printf("Completed.\n");
    printf("Memory usage (MB): %.2f\n", getMemoryUsage()/1024.0);

    printf("[Summ] %d %d %d %d %f\n",  omp_get_max_threads(), ny, nz, state.get_nv(), stepms/END_STEP/size*1000);
    return 0;
}
