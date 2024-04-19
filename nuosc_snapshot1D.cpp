

#include "nuosc_class.h"

// Get coarse-grained v-index: just evenly spread nv_target pts over nv_in point..
std::vector<int> gen_skimmed_vslice_index(uint sv, uint nv) {

    std::vector<int> v_slices(sv);
    uint dsv = std::floor(nv / (sv-1));
    for(int v=0;v<sv-1;v++) v_slices[v] = v*dsv;
    v_slices[sv-1] = nv - 1;
    return v_slices;
}

void NuOsc::addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> vidx)  {
#ifdef NVTX
    nvtxRangePush(__FUNCTION__);
#endif

    SnapShot ss(var, fntpl, dumpstep, vidx);
    snapshots.push_back(ss);
    int sv  = vidx.size();

    printf("Add %d x %d x %d (ZxPxV) snapshot every %d steps.\n", nz, 1, sv, dumpstep);

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< nz <<" "<< -1 << " "<< sv << endl;
    outfile << z0 <<" "<< z1 << endl;

    for (int i=0;i<nz; ++i) {
        outfile << Z[i]  << " ";
    }   outfile << endl;
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

        printf("		Writing %d vars of size %d x %d x %d (ZxPxV) into %s\n", ss.var_list.size(), nz,1, sv, filename);

        outfile.write((char *) &t,        sizeof(uint) );
        outfile.write((char *) &phy_time, sizeof(real) );

        std::vector<real> carr(nz*sv);
        for (auto const& var : ss.var_list) {

            #pragma omp parallel for collapse(2)
            //#pragma acc parallel loop collapse(2)
            for(int i=0; i<nz; ++i)
            for(int v=0; v<sv; ++v) {
                carr[ i*sv + v ] = var[ idx(0,i,vc[v]) ];
            }

            outfile.write((char *) carr.data(),  nz*sv*sizeof(real));
        }
        outfile.close();
    }
#ifdef NVTX
    nvtxRangePop();
#endif
}


void NuOsc::Dump2Text(char * fname) {

    std::ofstream outfile;
    outfile.open( fname, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  fname << endl;


    for (int k=0;k<nz; ++k)
        outfile << Z[k] << " ";
        outfile << endl;

      for (int v=0;v<nv; ++v)
        outfile << vz[v] << " ";
        outfile << endl;

    outfile.precision( std::numeric_limits<real>::digits10 + 1 );

    std::vector<real*> ff = {v_stat->ee,
            v_stat->xx,
            v_stat->ex_re,
            v_stat->ex_im,
            v_stat->bee,
            v_stat->bxx,
            v_stat->bex_re,
            v_stat->bex_im };


    for (int f=0; f<8;++f) {
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    for (int i=0;i<1; ++i)
    for (int j=0;j<nz; ++j) {
    for (int v=0;v<nv; ++v) {
        outfile << ff[f][idx(i,j,v)] << " ";
    }
        outfile << endl;
  }
        outfile << endl;
    }

    outfile.close();

}

