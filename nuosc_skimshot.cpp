#include "nuosc_class.h"

// Find the (vy,vz) index from (~0, -1) to (~0, 1) 
// Just a quick workaround to get coarse-grained v grid for snapshot from 2D v list.
vector<int> gen_skimmed_vslice_index(uint nv_target, uint nv_in, const real * vy, const real * vz) {

    assert(nv_in%2==0);
    vector<int> v_slices;
    real dv = 2.0/nv_in;
    uint v0, v1;
    for(int v=0; v<nv_in*nv_in;v++)  {
        //cout<< "Processing ... " << vy[v] << " " << vz[v] << endl; 
	if ( std::abs(vy[v] - 0.5*dv) < 0.1*dv  ) { v0=v; break; }
    }
    for(int v=v0;v<nv_in*nv_in;v++) {
        //cout<< "Processing ... " << vy[v] << " " << vz[v] << endl; 
	if ( std::abs(vy[v]-0.5*dv)> 0.5*dv ) { v1=v; break; }
    }
    
    uint dsv = ceil(float(v1-v0) / nv_target);
    
    for (int v=0;v<nv_target-1; v++)  v_slices.push_back(v0+dsv*v);
    v_slices.push_back(v1-1);

    printf("   Gen %d skimmed v-pts near vy= %f and vz in [ %f %f ].\n", 
		v_slices.size(), dv,
		vz[ v_slices[0] ], 
		vz[ v_slices.back() ] );
    
    return v_slices;
}

// Get coarse-grained v-index: just evenly spread nv_target pts over nv_in point. 
vector<int> gen_skimmed_vslice_index(uint sv, uint nv) {

    vector<int> v_slices(sv);
    uint dsv = floor(nv / (sv-1));

    for(int v=0;v<sv-1;v++) v_slices[v] = v*dsv;
  
    v_slices[sv-1] = nv - 1;
    
    return v_slices;
}

// Register skimmed shot for reduce sy, sz at selected v[v_idx], for both 1D/2D.
void NuOsc::addSkimShot(std::list<real*> var, char *fntpl, int dumpstep, int sy, int sz, vector<int> vidx) {
    //for (auto&i:vidx)  cout << i << " " ;

    SkimShot ss(var, fntpl, dumpstep, sy, sz, vidx);
    skimshots.push_back(ss);

    int dsz = floor(nz/sz);
    int dsy = floor(ny/sy);
    int sv  = vidx.size();

    printf("Add %d x %d x %d skimshot every %d steps. ( Simulation size %d x %d x %d )\n", sy, sz, sv, dumpstep, ny, nz, nv);
    //cout << "      at vz-slices: "; for(auto &v:vidx) printf("%g ",vz[v]); cout << endl;
    assert(nz>=sz && ny>=sy && nv>=sv);

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

    for(int i=0;i<ny; i+=dsy)   outfile << Y[i]  << " ";   outfile << endl;
    for(int j=0;j<nz; j+=dsz)   outfile << Z[j]  << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vy[v] << " ";   outfile << endl;
    for(auto &v:vidx)       outfile << vz[v] << " ";   outfile << endl;
}

// Take skimmed shot for addYZSkimShot()
void NuOsc::takeSkimShot(const int t) {

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

        printf("		Writing %d vars of size %d x %d x %d into %s\n", ss.var_list.size(), sy, sz, sv, filename);
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
        outfile.close();
    }
}

// NOT USED..
void NuOsc::takeSkimShotToConsole(const int t) {

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
