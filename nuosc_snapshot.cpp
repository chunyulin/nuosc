#include "nuosc_class.h"

// Get coarse-grained v-index: just evenly spread nv_target pts over nv_in point..
std::vector<int> gen_skimmed_vslice_index(uint sv, uint nv) {

    std::vector<int> v_slices(sv);
    uint dsv = std::floor(nv / (sv-1));
    for(int v=0;v<sv-1;v++) v_slices[v] = v*dsv;
    v_slices[sv-1] = nv - 1;
    return v_slices;
}

void NuOsc::addSnapShotAtXV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> xidx, std::vector<int> vidx)  {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    SnapShot ss(var, fntpl, dumpstep, xidx, vidx);
    snapshots.push_back(ss);
    int sx  = xidx.size();
    int sv  = vidx.size();

    if (myrank==0) printf("Add %d x %d x %d (XxZxV) snapshot every %d steps.\n", sx, nz, sv, dumpstep);

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< sx <<" "<< nz << " "<< sv << endl;
    outfile << x0 <<" "<< x1 << endl;
    outfile << z0 <<" "<< z1 << endl;

    for(auto &x:xidx)       outfile << X[x]  << " ";   outfile << endl;
    for (int i=0;i<nz; ++i) outfile << Z[i]  << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vx[v] << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vz[v] << " ";   outfile << endl;
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> vidx)  {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    SnapShot ss(var, fntpl, dumpstep, vidx);
    snapshots.push_back(ss);
    int sv  = vidx.size();

    if (myrank==0) printf("Add %d x %d x %d (XxZxV) snapshot every %d steps.\n", nx, nz, sv, dumpstep);

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< nx <<" "<< nz << " "<< sv << endl;
    outfile << x0 <<" "<< x1 << endl;
    outfile << z0 <<" "<< z1 << endl;

    for (int i=0;i<nx; ++i) {
        outfile << X[i]  << " ";
    }   outfile << endl;
    for (int i=0;i<nz; ++i) {
        outfile << Z[i]  << " ";
    }   outfile << endl;
    for(auto &v:vidx)       outfile << vx[v] << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vz[v] << " ";   outfile << endl;
#ifdef NVTX
    nvtxRangePop();
#endif
}

void NuOsc::checkSnapShot(const int t) const {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    for (auto const& ss : snapshots) {

        if ( t % ss.every != 0 ) break;

        std::vector<int> xc = ss.x_slices;
        int sx = xc.size();
        std::vector<int> vc = ss.v_slices;
        int sv = vc.size();

        char filename[32];
        sprintf(filename, ss.fntpl.c_str(), t);
        std::ofstream outfile;
        outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
        if(!outfile) cout << "*** Open fails: " <<  filename << endl;

        printf("		Writing %d vars of size %d x %d x %d (XxZxV) into %s\n", ss.var_list.size(), sx, nz, sv, filename);

        outfile.write((char *) &t,        sizeof(uint) );
        outfile.write((char *) &phy_time, sizeof(real) );

        std::vector<real> carr(nz*nx*sv);
        for (auto const& var : ss.var_list) {

            #pragma omp parallel for collapse(3)
            //#pragma acc parallel loop collapse(2)
            for(int i=0; i<sx; ++i)
            for(int j=0; j<nz; ++j)
            for(int v=0; v<sv; ++v) {
                carr[ (i*nz + j)*sv + v ] = var[ idx(xc[i],j,vc[v]) ];
            }

            outfile.write((char *) carr.data(),  sx*nz*sv*sizeof(real));
        }
        outfile.close();
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

