#include "nuosc_class.h"

// 2D y-z plane in real space
void NuOsc::addYZSkimShot(std::list<real*> var, char *fntpl, int dumpstep, int sy, int sz, std::vector<int> vidx) {
    SkimShot ss(var, fntpl, dumpstep, sy, sz, vidx);
    skimshots.push_back(ss);

    int sv = vidx.size();
    int dsz = nz/sz;
    int dsy = ny/sy;
    printf("   Add %dx%d YZ skimshot at %d v-slices: ", sy, sz, sv);
    for(auto &v:vidx) printf("(%g,%g) ",vy[v],vz[v]); cout << endl;

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< sy <<" "<< sz <<" "<< sv << endl;
    outfile << ny <<" "<< nz <<" "<< nv            << endl;
    outfile << y0 <<" "<< y1 <<" "<< z0 <<" "<< z1 << endl;

    for(int i=0;i<ny; i+=dsz)   outfile << Y[i]  << " ";   outfile << endl;
    for(int j=0;j<nz; j+=dsz)   outfile << Z[j]  << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vy[v] << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vz[v] << " ";   outfile << endl;
}

// Skimmed output to binaries file for post-processing
void NuOsc::checkYZSkimShot(const int t) {

    for (auto const& ss : skimshots) {

        std::vector<int> vc = ss.v_slices;
        int sy = ss.sy;
        int sz = ss.sz;
        int sv = vc.size();
        int dsy = ny/sy;
        int dsz = nz/sz;

        if ( t % ss.every != 0 ) break;

        char filename[32];
        sprintf(filename, ss.fntpl.c_str(), t);
        std::ofstream outfile;
        outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
        if(!outfile) cout << "*** Open fails: " <<  filename << endl;

        std::vector<real> carr(sy*sz*sv);

        for (auto const& var : ss.var_list) {

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
            for(int i=0; i<sy; i++)
                for(int j=0; j<sz; j++)
                    for(int v=0; v<sv; v++) {
                        carr[ (i*sz + j) * sv + v ] = var[ idx(i*dsy,j*dsz,vc[v]) ];
                    }

            outfile.write((char *) &t,        sizeof(uint) );
            outfile.write((char *) &phy_time, sizeof(real) );
            outfile.write((char *) carr.data(),  sy*sz*sv*sizeof(real));
        }
        printf("		Write %d vars of size %dx%dx%d into %s\n", ss.var_list.size(), sy, sz, sv, filename);
        outfile.close();
    }
}

// Skimmed output to binaries file for post-processing
void NuOsc::checkSkimShot(const int t) {

    for (auto const& ss : skimshots) {

        std::vector<int> vc = ss.v_slices;
        std::vector<int> yc = ss.y_slices;
        int sy = yc.size();
        int sz = ss.sz;
        int sv = vc.size();
        int dsz = nz/sz;

        if ( t % ss.every != 0 ) break;

        char filename[32];
        sprintf(filename, ss.fntpl.c_str(), t);

        std::ofstream outfile;
        outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
        if(!outfile) cout << "*** Open fails: " <<  filename << endl;

        std::vector<real> carr(sy*sz*sv);

        for (auto const& var : ss.var_list) {

#pragma omp parallel for collapse(3)
#pragma acc parallel loop collapse(3)
            for(int i=0; i<sy; i++)
                for(int j=0; j<sz; j++)
                    for(int v=0; v<sv; v++) {
                        carr[ (i*sz + j) * sv + v ] = var[ idx(yc[i],j*dsz,vc[v]) ];
                    }

            outfile.write((char *) carr.data(),  sy*sz*sv*sizeof(real));
        }
        printf("		Write %d vars of size %dx%dx%d into %s\n", ss.var_list.size(), sy, sv, sz, filename);
        outfile.close();
    }
}

// Skimmed output to console for debugging
void NuOsc::checkSkimShotToConsole(const int t) {

    for (auto const& ss : skimshots) {

        std::vector<int> vc = ss.v_slices;
        std::vector<int> yc = ss.y_slices;
        int sy = yc.size();
        int sz = ss.sz;
        int sv = vc.size();
        int dsz = nz/sz;

        if ( t % ss.every != 0 ) break;

        std::vector<real> carr(sy*sz*sv);

        for (auto const& var : ss.var_list) {

            for(int i=0; i<sy; i++)
            for(int v=0; v<sv; v++) {
        	printf("At v=%g y=%g: ", vz[vc[v]], Y[yc[i]]);
            
	        for(int j=0; j<sz; j++) cout << var[ idx(yc[i],j*dsz,vc[v]) ] << " ";
	        cout << endl;
            }
        }
    }
}

// not used 
void NuOsc::addSkimShot(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> y_slices, int sz, int sv) {
    SkimShot ss(var, fntpl, dumpstep, y_slices, sz, sv);
    skimshots.push_back(ss);

    auto dsz = nz/sz;
    auto dsv = (nv-1)/(sv-1);
    printf("   Add SkimShot with z: %d / %d / %d   v: %d / %d / %d   at y-slices ", nz, sz, dsz, nv, sv, dsv);
    for(auto &i:y_slices) cout << i << " ";  cout << endl;
    assert(nz%sz==0);
    assert((nv-1)%(sv-1)==0);


    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    for(auto &i:y_slices)  outfile << i << " ";   outfile << endl;
    outfile << nz << " " << sz << " " << dsz << " " << z0<< " " << z1 << " " << dt << endl;
    outfile << nv << " " << sv << " " << dsv << endl;

    for(int i=0;i<ny; i++)      outfile << Y[i] << " ";
    outfile << endl;
    for(int j=0;j<nz; j+=dsz)   outfile << Z[j] << " ";
    outfile << endl;
    for(int v=0;v<nv; v+=dsv)   outfile << vz[v] << " ";
    outfile << endl;

# if 0
    // mixed ascii and binary not working
    std::vector<real> tmpz(sz);
    for(int j=0;j<nz; j+=dsz)   tmpz[j/dsz] = Z[j];
    outfile.write((char *) tmpz.data(), sz*sizeof(real));
#endif
}

// 2D Generalization of the previous one that is mainly for 1D. Here we make nv into explicit v_slices...
void NuOsc::addSkimShot(std::list<real*> var, char *fntpl, int dumpstep, vector<int> y_slices, int sz, vector<int> v_slices) {
    SkimShot ss(var, fntpl, dumpstep, y_slices, sz, v_slices);
    skimshots.push_back(ss);

    auto dsz = nz/sz;
    auto sy = y_slices.size();
    auto sv = v_slices.size();
    printf("   Add SkimShot with z: %d / %d / %d  at %d v-slices and %d y-slices.\n", nz, sz, dsz, sv, sy);

    std::ofstream outfile;
    char filename[32];
    string tmp = string(fntpl) + ".meta";
    sprintf(filename, tmp.c_str(), 0);
    outfile.open( filename, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  filename << endl;

    // grid information
    outfile << dt <<" "<< sy <<" "<< sz <<" "<< sv << endl;
    outfile << ny <<" "<< nz <<" "<< nv            << endl;
    outfile << y0 <<" "<< y1 <<" "<< z0 <<" "<< z1 << endl;

    for(auto &i:y_slices)       outfile << Y[i] << " ";   outfile << endl;
    for(int j=0;j<nz; j+=dsz)   outfile << Z[j] << " ";   outfile << endl;
    for(auto &v:v_slices)       outfile << vz[v] << " ";   outfile << endl;

}


