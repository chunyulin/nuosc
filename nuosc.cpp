#include "nuosc_class.h"
#include "utils.h"
#include <limits>

// TODO: Handling OpenACC error.
typedef void (*exitroutinetype)(char *err_msg);
void acc_set_error_routine(exitroutinetype callback_routine);
void handle_gpu_errors(char *err_msg) {
    std::cout << "GPU Error: " << err_msg << std::endl;
    std::cout << "Exiting..." << std::endl << std::endl;
    #ifdef COSENU_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    #endif
    exit(-1);
}

int main(int argc, char *argv[]) {

    // Timing utilities
    bool is_restart = 0;
    int restart_from = 0;

    // Timing utilities
    float stepms;
    float stepms_max, stepms_min;
    #ifdef COSENU_MPI
    float t1;
    #else
    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    #endif

    // Nuosc parameters
    int px[DIM];
    real bbox[DIM][2];
    real dx = 0.1;
    for (int d=0; d<DIM; ++d) {
        px[d] = 1;
        bbox[d][0] = -0.2; bbox[d][1] = 0.2;
    }
    bbox[DIM-1][0] = -1; bbox[DIM-1][1] = 1;

    int nv_in = 8, nphi = 8;
    real cfl = 0.5;      real ko = 0.0;

    real mu  = 1.0;
    real pmo = 1.0;
    bool renorm = false;

    // === initial value
    real alpha = 0.9;                   // nuebar-nue asymmetric parameter
    real lnue [] = {std::numeric_limits<real>::max(), std::numeric_limits<real>::max(), 0.6};
    real lnueb[] = {std::numeric_limits<real>::max(), std::numeric_limits<real>::max(), 0.5};

    real ipt   = 0;                     // 0: central_z_perturbation; 1:random; 4:noc case
    real eps0  = 0.1;
    real sigma  = 100.0;    // lzpt = 2*simga**2

    int ANAL_EVERY = 5;    // 10.0  / (cfl*dz) + 1;
    int END_STEP   = 5;    // 900.0 / (cfl*dz) + 1;
    int DUMP_EVERY = 99999999;

    int ranks = 1, myrank = 0;

    #ifdef COSENU_MPI
    // THINK: consider to initialze MPI inside main class, may need passing argc argv into.
    int provided;
    // Thread support:  MPI_THREAD_SINGLE < MPI_THREAD_FUNNELED <  MPI_THREAD_SERIALIZED < MPI_THREAD_MULTIPLE
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (!myrank) printf("[%.4f] MPI_Init_thread mode: %d\n", utils::msecs_since(), provided);
    #endif

    // Parse input argument --------------------------------------------
    for (int t = 1; argv[t] != 0; t++) {
        if (strcmp(argv[t], "--dx") == 0 )  {
            dx = atof(argv[t+1]);     t+=1;
        } else if (strcmp(argv[t], "--xmax") == 0 )  {
            for (int d=0; d<DIM; ++d) {
                bbox[d][1] = atof(argv[t+1]);    t+=1;
                bbox[d][0] = -bbox[d][1];
            }
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
            if (!myrank) cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY") == 0 )  {
            DUMP_EVERY = atoi(argv[t+1]);    t+=1;
            if (!myrank) cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--END_STEP") == 0 )  {
            END_STEP = atoi(argv[t+1]);    t+=1;
            if (!myrank) cout << " ** END_STEP: " << END_STEP << endl;
        } else if (strcmp(argv[t], "--ANA_EVERY_T") == 0 )  {
            ANAL_EVERY = int( atof(argv[t+1]) / (cfl*dx) + 0.5 );    t+=1;
            if (!myrank) cout << " ** ANAL_EVERY: " << ANAL_EVERY << endl;
        } else if (strcmp(argv[t], "--DUMP_EVERY_T") == 0 )  {
            DUMP_EVERY = int ( atof(argv[t+1]) / (cfl*dx) + 0.5 );    t+=1;
            if (!myrank) cout << " ** DUMP_EVERY: " << DUMP_EVERY << endl;
        } else if (strcmp(argv[t], "--END_STEP_T") == 0 )  {
            END_STEP = int ( atof(argv[t+1]) / (cfl*dx) + 0.5 );    t+=1;
            if (!myrank) cout << " ** END_STEP: " << END_STEP << endl;
            // for intial data
        } else if (strcmp(argv[t], "--lnue") == 0 )  {
            lnue[2]   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnueb") == 0 )  {
            lnueb[2]  = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnuex") == 0 )  {
            lnue[0]   = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--lnuebx") == 0 )  {
            lnueb[0]  = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--sigma") == 0 )  {
            sigma = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--eps0") == 0 )  {
            eps0 = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--alpha") == 0 )  {
            alpha = atof(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--ipt") == 0 )  {
            ipt = atoi(argv[t+1]);    t+=1;
        } else if (strcmp(argv[t], "--np") == 0 )  {
            for (int d=0; d<DIM; ++d) { px[d] = atoi(argv[t+1]); t+=1; }
        } else if (strcmp(argv[t], "--cpt") == 0 )  {
            is_restart = 1;
            restart_from = atoi(argv[t+1]);  t+=1;
        } else {
            printf("Unreconganized parameters %s!\n", argv[t]);
            exit(0);
        }
    }

#ifdef _OPENACC
    //acc_set_error_routine(&handle_gpu_errors);  // undefined 
#endif

#if defined(SCHEME_WENO7) || defined(SCHEME_FD8)
    #if   DIM == 2
    int gx[] = {4,4};
    #elif DIM == 3
    int gx[] = {4,4,4};
    #endif
#elif !defined(KO_ORD_3)
    #if   DIM == 2
    int gx[] = {3,3};
    #elif DIM == 3
    int gx[] = {3,3,3};
    #endif
#else
    #if   DIM == 2
    int gx[] = {2,2};
    #elif DIM == 3
    int gx[] = {2,2,2};
    #endif
#endif

    // === create simuation
    NuOsc state(px, nv_in, nphi, gx, bbox, dx, cfl, ko);
    if (!myrank) printf("[%.4f] Initialize main class.\n", utils::msecs_since());

    auto lpts = state.get_lpts();
    state.set_mu(mu);
    state.set_pmo(pmo);
    state.set_renorm(renorm);

    uint nx[DIM];
    for (int d=0; d<DIM; ++d) nx[d] = (bbox[d][1]-bbox[d][0])/dx;

#ifdef ADV_TEST
    if      (ipt==10) state.fillInitGaussian( eps0, sigma);
    else if (ipt==20) state.fillInitSquare( eps0, sigma);
    else if (ipt==30) state.fillInitTriangle( eps0, sigma);
#else
    if (is_restart)   state.restoreInitValue(restart_from, alpha, lnue, lnueb);
    else              state.fillInitValue(ipt, alpha, eps0, sigma, lnue, lnueb);
#endif
    if (!myrank) printf("[%.4f] Initialize data done.\n", utils::msecs_since());

    // === init analysis
    state.analysis();
    if (!myrank) printf("[%.4f] First analysis done.\n", utils::msecs_since());

/*
    if (DUMP_EVERY <= END_STEP) {
#ifdef ADV_TEST
        std::list<std::vector<real>> vlist( { state.v_stat->wf[ff::ee] } );
        state.addSnapShotAtV(vlist, "ee%06d.bin", DUMP_EVERY,  std::vector<int>{0,state.get_nv()/2, state.get_nv()-1} );
        //state.addSnapShotAtV(vlist, "ee%06d.bin", DUMP_EVERY, gen_skimmed_vslice_index(nv_in, nv_in)  );
#else
        //state.addSnapShotAtXV(vlist, "P3_%06d.bin", DUMP_EVERY, std::vector<int>{0,nx[0]/2,nx[0]-1}, vslice );
        //std::list<real*> plist( { state.P3 } );
        //state.addSkimShot(plist, "P3_%06d.bin", DUMP_EVERY, nz, 11 );
        //std::list<real*> rlist( {state.v_stat->ee, state.v_stat->xx} );
        //state.addSkimShot(rlist, "Rho%06d.bin", DUMP_EVERY, 10240, 21 );

#endif
        //if (!is_restart) state.checkSnapShot();
        //state.checkSkimShots();
    }
*/
    { // prepare checkpoint
    std::list<int>   vlist;  for (int f=0; f<state.nvar; ++f) vlist.push_back(f);
    std::vector<int> vslice; for (int v=0;v<state.nv;++v)     vslice.push_back(v);
    state.addSnapShotAtV("ckp", vlist, DUMP_EVERY, vslice );
    }

    if (!myrank) std::cout << std::flush;

    const int cooltime = 3;
    for (int t=1; t<=END_STEP; t++) {
        #ifdef COSENU_MPI
        if (t==cooltime)  t1 = MPI_Wtime();
        #else
        if (t==cooltime)  t1 = std::chrono::high_resolution_clock::now();
        #endif
        state.step_rk4();

        if ( t%ANAL_EVERY==0)  {
            //state.fft();
            state.analysis();
        }

        state.checkSnapShot();

        if ( t==10 || t==100 || t==1000 || t==END_STEP || state.stop_flag) {
            #ifdef COSENU_MPI
            stepms = (MPI_Wtime() - t1)*1e3;
            MPI_Reduce(&stepms, &stepms_max, 1, MPI_FLOAT, MPI_MAX, 0, state.CartCOMM);
            MPI_Reduce(&stepms, &stepms_min, 1, MPI_FLOAT, MPI_MIN, 0, state.CartCOMM);
            #else
            stepms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count();
            stepms_max = stepms_min = stepms;
            #endif
            if (myrank==0) {
               stepms_min /= state.ssize;
               stepms_max /= state.ssize;
               printf("%d Walltime: (Min) %.3f s/T, %.2f ns/step-grid.    (Max) %.3f s/T, %.2f ns/step-grid.\n", t,
               stepms_min/state.phy_time/1000,  stepms_min/(t-cooltime+1)/lpts*1e6,
               stepms_max/state.phy_time/1000,  stepms_max/(t-cooltime+1)/lpts*1e6 );
               fflush(stdout);
            }
        }

        if (state.stop_flag) break;

    }  // end of main loop

    // Get total memory
    float tmem = utils::getMemoryUsage()/1024./1024;
    float tmem_max = tmem, tmem_min = tmem;
    #ifdef COSENU_MPI
    MPI_Reduce(&tmem, &tmem_max, 1, MPI_FLOAT, MPI_MAX, 0, state.CartCOMM);
    MPI_Reduce(&tmem, &tmem_min, 1, MPI_FLOAT, MPI_MIN, 0, state.CartCOMM);
    #endif
    if (myrank==0) {
       #ifdef _OPENMP
       int tids = omp_get_max_threads();
       #else
       int tids = 1;
       #endif
       double ns_per_stepgrid = stepms_max/(END_STEP-cooltime+1)/lpts*1e6;
       double s_per_phytime   = stepms_max/state.phy_time/1000;
       printf("Completed.\n\n");
       printf("Memory usage (GB) per rank: %.2f ~ %.2f\n", tmem_min, tmem_max );
       printf("[Summ] %d %d %d %d %d %d %d %d %f %f\n", tids, px[0],px[1],px[2], nx[0],nx[1],nx[2], state.get_nv(), ns_per_stepgrid, s_per_phytime);
    }
    #ifdef PROFILE
    state.profile << "Memory GB " << tmem << endl;
    state.profile << "ms_per_step_grid: " << stepms/(END_STEP-cooltime+1)/lpts*1e6 << " " << stepms/state.phy_time/1000 << endl;
    #endif

    #ifdef SYNC_NCCL
    ncclCommDestroy(state._ncclcomm);   // FIXME: should redesign class moving this to another place
    #endif
    #if defined(COSENU_MPI)
    MPI_Finalize();    // Why segfault?
    #endif
    return 0;
}
