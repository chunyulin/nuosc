#include "nuosc_class.h"

// Get coarse-grained v-index: just evenly spread nv_target pts over nv_in point..
std::vector<int> gen_skimmed_vslice_index(uint sv, uint nv) {

    std::vector<int> v_slices(sv);
    uint dsv = std::floor(nv / (sv-1));
    for(int v=0;v<sv-1;v++) v_slices[v] = v*dsv;
    v_slices[sv-1] = nv - 1;
    return v_slices;
}

/*
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
*/

void NuOsc::addSnapShotAtV(std::list<std::vector<real>> var, char *fntpl, int dumpstep, std::vector<int> vidx)  {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    SnapShot ss(var, fntpl, dumpstep, vidx);
    snapshots.push_back(ss);
    int sv  = vidx.size();

    if (myrank==0) printf("Add %d x %d x %d (XxZxV) snapshot every %d steps.\n", nx[0], nx[2], sv, dumpstep);

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< nx[0] <<" "<<  nx[2] << " "<< sv << endl;
    outfile << bbox[0][0] <<" "<< bbox[0][1] << endl;
    outfile << bbox[2][0] <<" "<< bbox[2][0] << endl;

    for (int i=0;i<nx[0]; ++i) {
        outfile << X[0][i]  << " ";  // X
    }   outfile << endl;
    for (int i=0;i<nx[2]; ++i) {
        outfile << X[2][i]  << " ";  // Z
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

        std::vector<int> vc = ss.v_slices;
        int sv = vc.size();

        char filename[32];
        sprintf(filename, ss.fntpl.c_str(), t);
        std::ofstream outfile;
        outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
        if(!outfile) cout << "*** Open fails: " <<  filename << endl;

        printf("		Writing %ld vars of size %d x %d x %d (XxZxV) into %s\n", ss.var_list.size(), nx[0], nx[2], sv, filename);

        outfile.write((char *) &t,        sizeof(uint) );
        outfile.write((char *) &phy_time, sizeof(real) );

        std::vector<real> carr(nx[2]*nx[0]*sv);
        for (auto const& var : ss.var_list) {

            #pragma omp parallel for collapse(3)
            //#pragma acc parallel loop collapse(2)
            for(int i=0; i<nx[0]; ++i)
            for(int k=0; k<nx[2]; ++k)
            for(int v=0; v<sv; ++v) {
                carr[ (i*nx[2] + k)*sv + v ] = var[ idx(i,1/* fix Y*/ ,k,vc[v]) ];
            }

            outfile.write((char *) carr.data(),  nx[0]*nx[2]*sv*sizeof(real));
        }
        outfile.close();
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}

