#include "nuosc_class.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// Get coarse-grained v-index: just evenly spread nv_target pts over nv_in point..
std::vector<int> gen_skimmed_vslice_index(uint sv, uint nv) {

    std::vector<int> v_slices(sv);
    uint dsv = std::floor(nv / (sv-1));
    for(int v=0;v<sv-1;v++) v_slices[v] = v*dsv;
    v_slices[sv-1] = nv - 1;
    return v_slices;
}

void NuOsc::addSnapShotAtV(string tag, std::list<int> var, int dumpstep, std::vector<int> vidx)  {
#ifdef PROFILE
    nvtxRangePush(__FUNCTION__);
#endif

    mkdir(CKPT.c_str(), 0700);

    SnapShot ss(tag, var, dumpstep, vidx);
    snapshots.push_back(ss);
    int sv  = vidx.size();

    if (!myrank) printf("Add %zu fields snapshot of [ %d x %d x %d x %d ] every %d steps.\n", var.size(), nx[0], nx[1], nx[2], sv, dumpstep);

    std::ofstream outfile;
    char filename[32];

    string tmp = CKPT + "/" + tag + ".meta." + std::to_string(myrank);
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< nx[0] <<" "<<nx[1] <<" "<<  nx[2] << " "<< sv << endl;
    for (int d=0;d<DIM;++d) {
       outfile << bbox[d][0] <<" "<< bbox[d][1] << endl;
       for (int i=0;i<nx[d]; ++i) outfile << X[d][i]  << " ";
       outfile << endl;
    }
    for(auto &v:vidx)  outfile << vx[v] << " " << vy[v] << " " << vz[v] << " " << endl;
#ifdef PROFILE
    nvtxRangePop();
#endif
}

void NuOsc::checkSnapShot() {
#ifdef PROFILE
    nvtxRangePush(__FUNCTION__);
#endif

    for (auto const& ss : snapshots) {

        if ( 0 != iter % ss.every ) break;

        std::vector<int> vc = ss.v_slices;
        int sv = vc.size();

        std::vector<real> carr(nx[0]*nx[1]*nx[2]*sv);
        for (auto const& var : ss.var_list) {

            string filename;
            filename = CKPT + "/" + ss.tag + std::to_string(var) + "_" + std::to_string(iter) + "." + std::to_string(myrank);
            std::ofstream outfile;
            outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
            if(!outfile) cout << "*** Open fails: " <<  filename << endl;

            if (!myrank) printf("	Writing fid:%d of [ %d x %d x %d x %d ] into %s\n", var, nx[0], nx[1], nx[2], sv, filename.c_str());

            outfile.write((char *) &iter,     sizeof(uint) );
            outfile.write((char *) &phy_time, sizeof(real) );

            PARFORALL(i,j,k,v) {
                carr[ v + sv*( k + nx[2]*( j + nx[1]*i)) ] = v_stat->wf[var][ idx(i,j,k,vc[v]) ];
            }
            outfile.write((char *) carr.data(),  nx[0]*nx[1]*nx[2]*sv*sizeof(real));
            outfile.close();
        }
    }
#ifdef PROFILE
    nvtxRangePop();
#endif
}