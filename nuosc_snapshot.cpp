#include "nuosc_class.h"

void NuOsc::addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> vidx)  {

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
}

void NuOsc::checkSnapShot(const int t) const {

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
        std::vector<real> carr(nz*sv);

        for (auto const& var : ss.var_list) {

#pragma omp parallel for collapse(2)
#pragma acc parallel loop collapse(2)
            for(int i=0; i<nz; ++i)
            for(int v=0; v<sv; ++v) {
                carr[ i*sv + v ] = var[ idx(0,i,vc[v]) ];
            }

            outfile.write((char *) &t,        sizeof(uint) );
            outfile.write((char *) &phy_time, sizeof(real) );
            outfile.write((char *) carr.data(),  nz*sv*sizeof(real));
        }
        outfile.close();
    }
}

