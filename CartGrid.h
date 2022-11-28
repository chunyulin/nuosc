#pragma once
#include "common.h"

//#define IM_V2D_POLAR_GL_Z
//#define IM_V2D_POLAR_RSUM   // uniform vz-/phi- as default

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
        real dx;            // dx, dy, dz
        int  px[DIM];       // processor geometry: px, py, pz
        int  nx[DIM];       // shape: nx, ny, nz
        int  gx[DIM];
        Vec  X[DIM];        // coordinates X,Y,Z
        real bbox[DIM][2];  // bounding box [x0,x1] [y0,y1] [z0,z1]

        int nv, nphi;       // # of v cubature points.
        Vec vz, vx, vy, vw; // v coors and integral quadrature

        ulong lpts;

        int ranks = 1, myrank = 0;
        int srank = 0;
        int rx[DIM] = {0};
        int nb[DIM][2] = {0};    // Cartensian neighbor ranks

        int ngpus = 0;
#ifdef COSENU_MPI
        MPI_Comm CartCOMM;

        // packing buffers for X- X+, Y- Y+, and Z- Z+.
        MPI_Datatype t_pb[DIM];
        MPI_Win      w_pb[DIM];

#endif

        real *pb[DIM];     // TODO: can we use the storage of FieldVar directly w/o copying to this buffers ?

        // calculates local box and cartesian communicator given global box and processor shape
        CartGrid(int px_[], int nv_, const int nphi_, const int gx_[], const real bbox_[][2], const real dx_);

        ~CartGrid();
        inline void print_info() {
            char hname[20];
            char tag[50];
            if(0!=gethostname(hname,sizeof(hname))) { cout << "Error in gethostname" << endl; }

            if (ngpus > 0) sprintf(tag, "GPU %d %s", srank%ngpus, hname);
            else           sprintf(tag, "%s", hname);

            printf("[ Rank %2d ( %d %d %d ) on %s ]  nber:( %d %d )( %d %d )( %d %d )  N:[ %d %d %d %d ](nphi:%d)   bbox:( %g %g )( %g %g )( %g %g ) ]\n",
            myrank, rx[0],rx[1],rx[2], tag, nb[0][0],nb[0][1],nb[1][0],nb[1][1],nb[2][0],nb[2][1], nx[0],nx[1],nx[2], nv,nphi, bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1]);
        }

        void sync_isend();
        void sync_put();
        void sync_copy();

        int  get_nv()    const {  return nv;   }
        int  get_nphi()  const  {  return nphi;   }
        ulong get_lpts()  const  {  return lpts;  }

#if DIM == 2
        inline ulong idx(const int i, const int j, const int k, const int v) const { return v + nv * ( (k+gx[2]) + (nx[1]+2*gx[1])*(  j+gx[1] )  ); }
#elif DIM == 3
        inline ulong idx(const int i, const int j, const int k, const int v) const { return v + nv * ( (k+gx[2]) + (nx[2]+2*gx[2])*( (j+gx[1])+ (nx[1]+2*gx[1])*(i+gx[0]) ) ); }
#endif

};
