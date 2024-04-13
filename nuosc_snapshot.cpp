#include "nuosc_class.h"
#include "utils.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <zlib.h>

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
    outfile << "## L1     : dt nx[0] nx[1] nx[2] nv"
            << "## L2,4,6 : bbox[d,0] bbox[d,1]"
            << "## L3,5,7 : X[d] coordinate"
            << "## L8-    : vgrid coordinate" << endl;
    
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

void NuOsc::checkSnapShot(bool init) {
#ifdef PROFILE
    nvtxRangePush(__FUNCTION__);
#endif

    const ulong wtime_limit_ms = wtime_limit_hour * 3600000;
    for (auto const& ss : snapshots) {

        if (wtime_limit_ms > 0 && utils::msecs_since() + stepms_max > wtime_limit_ms) {
            if (!myrank) printf("Checkpoint and simulation stop due to walltime limits.\n");
            stop_flag = 1;
        } else if (0 != iter%ss.every) continue;
        std::vector<int> vc = ss.v_slices;
        int sv = vc.size();

        std::vector<real> carr(nx[0]*nx[1]*nx[2]*sv);

        // dump time-independent ELN at first
        if (init) {
            string fname = CKPT + "/eln." + std::to_string(myrank);
            gzFile fp = gzopen(fname.c_str(),"wb");
            PARFORALL(i,j,k,v) {
                carr[ v + sv*( k + nx[2]*( j + nx[1]*i)) ] = G0[ idx(i,j,k,vc[v]) ];
            }
            gzwrite(fp, (void*) (carr.data()), nx[0]*nx[1]*nx[2]*sv*sizeof(real));
            PARFORALL(i,j,k,v) {
                carr[ v + sv*( k + nx[2]*( j + nx[1]*i)) ] = G0b[ idx(i,j,k,vc[v]) ];
            }
            gzwrite(fp, (void*) (carr.data()), nx[0]*nx[1]*nx[2]*sv*sizeof(real));
            gzclose(fp);
        }

        // Then, process each sanpshot
        string fo = CKPT+"/it"+std::to_string(iter);
        mkdir(fo.c_str(), 0700);
        for (auto const& var : ss.var_list) {

            string fname = fo + "/" + ss.tag + std::to_string(var) + "." + std::to_string(myrank);

            #ifdef NOCOMPRESS
            std::ofstream outfile( fname, std::ofstream::out | std::ofstream::trunc);
            outfile.write((char *) &iter,     sizeof(uint) );
            outfile.write((char *) &phy_time, sizeof(real) );
            #else
            gzFile fp = gzopen(fname.c_str(),"wb");
            gzwrite(fp, (char*)&iter    , sizeof(uint));
            gzwrite(fp, (char*)&phy_time, sizeof(real));
            #endif
            //if (!myrank) printf("	Writing fid:%d of [ %d %d %d %d ] into %s\n", var, nx[0], nx[1], nx[2], sv, filename.c_str());

            #pragma omp parallel for collapse(4)
            #pragma acc parallel loop collapse(4)
            for(int i=0; i<nx[0]; ++i)
            for(int j=0; j<nx[1]; ++j)
            for(int k=0; k<nx[2]; ++k)
            for(int v=0; v<sv; ++v) {
                carr[ v + sv*( k + nx[2]*( j + i*nx[1])) ] = v_stat->wf[var][ idx(i,j,k,vc[v]) ];
            }

            #ifdef NOCOMPRESS
            outfile.write((char *) carr.data(), nx[0]*nx[1]*nx[2]*sv*sizeof(real) );
            outfile.close();
            #else
            gzwrite(fp, (char*) carr.data(), nx[0]*nx[1]*nx[2]*sv*sizeof(real));
            gzclose(fp);
            #endif
        }
    }
#ifdef PROFILE
    nvtxRangePop();
#endif
}

void NuOsc::Dump2Text(char * fname) {

    std::ofstream outfile;
    outfile.open( fname, std::ofstream::out | std::ofstream::trunc);
    if(!outfile) cout << "*** Open fails: " <<  fname << endl;

    for (int k=0;k<nx[2]; ++k)
        outfile << X[DIM-1][k] << " ";
        outfile << endl;

      for (int v=0;v<nv; ++v)
        outfile << vz[v] << " ";
        outfile << endl;

    outfile.precision( std::numeric_limits<real>::digits10 + 1 );

    std::vector<real*> ff = {v_stat->wf[ff::ee],
            v_stat->wf[ff::mm],
            v_stat->wf[ff::emr],
            v_stat->wf[ff::emi],
            v_stat->wf[ff::bee],
            v_stat->wf[ff::bmm],
            v_stat->wf[ff::bemr],
            v_stat->wf[ff::bemi] };


    for (int f=0; f<8;++f) {
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    outfile << "# " <<f << endl;
    for (int i=0;i<nx[0]; ++i)
    for (int j=0;j<nx[1]; ++j)
    for (int k=0;k<nx[2]; ++k) {
      for (int v=0;v<nv; ++v) {
        outfile << ff[f][ idx(i,j,k,v)] << " ";
      }
        outfile << endl;
    }
        outfile << endl;
    }

    outfile.close();

}