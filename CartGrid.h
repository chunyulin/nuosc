#pragma once
#include "common.h"

//#define IM_V2D_POLAR_GL_Z
//#define IM_V2D_POLAR_RSUM

//#define IM_SIMPSON
//#define IM_TRAPEZOIDAL
//#define IM_GL

int gen_v2d_GL_zphi(const int nvz_, const int nphi_, Vec& vw, Vec& vx, Vec& vy, Vec& vz);
int gen_v2d_rsum_zphi(const int nvz_, const int nphi_, Vec& vw, Vec& vx, Vec& vy, Vec& vz);

int gen_v1d_GL(const int nv, Vec vw, Vec vz);
int gen_v1d_trapezoidal(const int nv_, Vec vw, Vec vz);
int gen_v1d_simpson(const int nv_, Vec vw, Vec vz);
int gen_v1d_cellcenter(const int nv_, Vec vw, Vec vz);

//
//  Holding the geometry and boundary packing buffer used by MPI
//
class CartGrid {

    public:
        int  nvar;
        int  px = 1, pz = 1;     // processor geometry: px, pz
        real z0, z1, x0, x1;  // bounding box [x0,x1] [z0,z1]
        real dx, dz;     // dx, dz
        int  nx, nz;     // shape: nx, nz
        int  gx, gz;
        Vec X, Z;           // coordinates

        int nv, nphi;       // # of v cubature points.
        Vec vz, vx, vy, vw; // v coors and integral quadrature

        ulong lpts;

        uint lpts_x;
        uint lpts_z;

        int lowerZ = 0, upperZ = 0, lowerX = 0, upperX = 0;

        int ranks=1, myrank=0;
        int srank = myrank;
        int rx = 0, rz = 0;

        int ngpus = 0;
#ifdef COSENU_MPI
        MPI_Comm CartCOMM;

        // packing buffers for Z- Z+ and X_- X+
        MPI_Datatype t_pbX, t_pbZ;
        MPI_Win      w_pbX, w_pbZ;

#endif

        real *pbX, *pbZ;          // TODO: can we use the storage of FieldVar directly w/o copying to this buffers ?

        // calculates local box and cartesian communicator given global box and processor shape
        CartGrid(int px_, int pz_, int nv_, int nphi_, int gx_, int gz_, real x0_, real x1_, real z0_, real z1_, real dx_, real dz_);
        ~CartGrid();

        inline void print_info() {
            char s[20] = {0};
            if(0!=gethostname(s,sizeof(s))) { cout << "Error in gethostname" << endl; }
            
            if (ngpus > 0)
              printf("[ Rank %2d on GPU-%d %s ]  My coor:( %d %d )  nbX:( %d %d ) nbZ:( %d %d )   Box: X ( %g %g ) %d  Z ( %g %g ) %d  V: %d %d\n",
              myrank, srank, s, rx, rz, lowerX, upperX, lowerZ, upperZ, x0, x1, nx, z0, z1, nz, nv, nphi);
            else
              printf("[ Rank %2d on %s ]  My coor:( %d %d )  nbX:( %d %d ) nbZ:( %d %d )   Box: X ( %g %g ) %d  Z ( %g %g ) %d  V: %d %d\n",
              myrank, s, rx, rz, lowerX, upperX, lowerZ, upperZ, x0, x1, nx, z0, z1, nz, nv, nphi);
        }

        void sync_buffer();
        void sync_buffer_isend();
        void sync_buffer_copy();

        int  get_nv()    const {  return nv;   }
        int  get_nphi()  const  {  return nphi;   }
        ulong get_lpts()  const  {  return lpts;  }
        uint  get_lpts_x()  const  {  return lpts_x;  }
        uint  get_lpts_z()  const  {  return lpts_z;  }

        inline ulong idx(const int i, const int j, const int v) const { return ( (i+gx)*(nz+2*gz) + j+gz)*nv + v; }    //  i:x j:z

/*
        void test_win() {
            const auto npbX = nvar*gx*nz*nv;
            for (int i=0;i<5;i++) pbX[i]= i+1;

            cout << "== Before == ";
            for (int i=0;i<10;i++) cout << pbX[i+2*npbX] << " ";
            MPI_Win_fence(0, w_pbX);
            MPI_Put(&pbX[0],    1, t_pbX, lowerX, 2, 1, t_pbX, w_pbX);
            MPI_Win_fence(0, w_pbX);
            cout << "== After == ";
            for (int i=0;i<10;i++) cout << pbX[i+2*npbX] << " ";
            cout <<endl;
        }
*/

};
