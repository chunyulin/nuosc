#pragma once
#include "common.h"

#define IM_V2D_POLAR_GL_Z
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
#ifdef COSENU_MPI
        MPI_Comm CartCOMM;

        // packing buffers for Z- Z+ and X_- X+
        MPI_Datatype t_pbX, t_pbZ;
        MPI_Win      w_pbX, w_pbZ;
#endif

        real *pbX, *pbZ;          // TODO: can we use the storage of FieldVar directly w/o copying to this buffers ?

        // calculates local box and cartesian communicator given global box and processor shape
        CartGrid(int px_, int pz_, int nv_, int nphi_, int gx_, int gz_, real x0_, real x1_, real z0_, real z1_, real dx_, real dz_) 
            : px(px_), pz(pz_), nphi(nphi_), gx(gx_), gz(gz_), dx(dx_), dz(dz_), nvar(8)
        {

            int ranks=1, myrank=0;
            int rx = 0, rz = 0;
#ifdef COSENU_MPI
            MPI_Comm_size(MPI_COMM_WORLD, &ranks);
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            // create Cartesian topology
            int periods[2] = {1,1};
            int pdims[2] = {px, pz};
            MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 0, &CartCOMM);

            // calcute local geometry from computational domain (x0_, x1_, z0_, z1_)
            int coords[2];
            MPI_Cart_coords(CartCOMM, myrank, 2, coords);
            rx = coords[0];
            rz = coords[1];

            MPI_Cart_shift(CartCOMM, 0, 1, &lowerX, &upperX);
            MPI_Cart_shift(CartCOMM, 1, 1, &lowerZ, &upperZ);
#endif

            x0 = x0_ + (x1_-x0_)*rx    /px;
            x1 = x0_ + (x1_-x0_)*(rx+1)/px;  if (rx+1 == px)  x1 = x1_;
            z0 = z0_ + (z1_-z0_)*rz    /pz;
            z1 = z0_ + (z1_-z0_)*(rz+1)/pz;  if (rz+1 == pz)  z1 = z1_;

            nx  = int((x1-x0)/dx);
            nz  = int((z1-z0)/dz);

            X.reserve(nx);   for(int i=0;i<nx; ++i)    X[i] = x0 + (i+0.5)*dx;
            Z.reserve(nz);   for(int i=0;i<nz; ++i)    Z[i] = z0 + (i+0.5)*dz;

#if defined(IM_V2D_POLAR_GL_Z)
            nv = gen_v2d_GL_zphi(nv_,nphi_, vw, vx, vy, vz);
#else
            nv = gen_v2d_rsum_zphi(nv_,nphi_, vw, vx, vy, vz);
#endif

            printf("[ Rank %2d ]  My coor:( %d %d )  nbX:( %d %d ) nbZ:( %d %d )   Box: X ( %g %g ) %d  Z ( %g %g ) %d  V: %d %d\n",
                    myrank, rx, rz, lowerX, upperX, lowerZ, upperZ, x0, x1, nx, z0, z1, nz, nv, nphi);

            lpts = (nx+2*gx)*(nz+2*gz)*nv;
            //lpts_x = gx*nz
            //lpts_z = gz*nx

            // prepare datatype for ghostzone block of each dimension
            const auto npbX = nvar*gx*nz*nv;
            const auto npbZ = nvar*nx*gz*nv;

#ifdef COSENU_MPI
            MPI_Type_contiguous(npbX, MPI_DOUBLE, &t_pbX);  MPI_Type_commit(&t_pbX);
            MPI_Type_contiguous(npbZ, MPI_DOUBLE, &t_pbZ);  MPI_Type_commit(&t_pbZ);

            // prepare (un-)pack buffer and MPI RMA window for sync. (duplicate 4 times for left/right and old/new)
            int ierr = 0;
            ierr = MPI_Win_allocate(4*npbX*sizeof(real), npbX*sizeof(real), MPI_INFO_NULL, CartCOMM, &pbX, &w_pbX);
            ierr = MPI_Win_allocate(4*npbZ*sizeof(real), npbZ*sizeof(real), MPI_INFO_NULL, CartCOMM, &pbZ, &w_pbZ);
            if (ierr!=MPI_SUCCESS) {  cout << " !! MPI error " << ierr << " at " << __FILE__ << ":" << __LINE__ << endl; }
#else
            pbX = new real[4*npbX];
            pbZ = new real[4*npbZ];
#endif
            #pragma acc enter data create(this,pbX[0:4*npbX],pbZ[0:4*npbZ])
            
            //test_win();


        }

        void test_win() {
            const auto npbX = nvar*gx*nz*nv;
            for (int i=0;i<5;i++) pbX[i]= i+1;

            cout << "== Before == ";
            for (int i=0;i<10;i++) cout << pbX[i+2*npbX] << " ";
            MPI_Win_fence(0, w_pbX);
            MPI_Put(&pbX[0],    1, t_pbX, lowerX, 2 /* skip old X block */, 1, t_pbX, w_pbX);
            MPI_Win_fence(0, w_pbX);
            cout << "== After == ";
            for (int i=0;i<10;i++) cout << pbX[i+2*npbX] << " ";
            cout <<endl;
        }


        ~CartGrid() {
            //MPI_Win_free(&w_pbX);   // will free by the MPI_Finalize() anyway.
            //MPI_Win_free(&w_pbZ);
            //MPI_Type_free(&t_pbX);
            //MPI_Type_free(&t_pbZ);

            #pragma acc exit data delete(pbX,pbZ)

#ifndef COSENU_MPI
            delete[] pbX;
            delete[] pbZ;
#endif
        }


        void sync_buffer() {
            const auto npbX = nvar*gx*nz*nv;
            const auto npbZ = nvar*nx*gz*nv;
#ifdef COSENU_MPI

#if 0
            for (int i=0;i<5;i++) { pbX[i]= i+1; pbX[i+npbX]= i+10; }
            for (int i=0;i<5;i++) { pbZ[i]= i+1; pbZ[i+npbZ]= i+10; }
            for (int i=0;i<5;i++) pbX[i+2*npbX]= 0;
            for (int i=0;i<5;i++) pbZ[i+2*npbZ]= 0;

            cout << "== BeforeX: ";
            for (int i=0;i<5;i++) {
              cout << pbX[i+2*npbX] << " ";
            }
            cout << "\t Z: ";
            for (int i=0;i<5;i++) {
              cout << pbZ[i+2*npbZ] << " ";
            } cout << endl;
#endif
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                #pragma acc update self( pbX[0:2*npbX] )
                MPI_Win_fence(0, w_pbX);
                MPI_Put(&pbX[0],    1, t_pbX, lowerX, 2 /* skip old X block */, 1, t_pbX, w_pbX);
                MPI_Put(&pbX[npbX], 1, t_pbX, upperX, 3 /* skip old X block */, 1, t_pbX, w_pbX);
                MPI_Win_fence(0, w_pbX);
                //#pragma acc update device( pbX[2*nvar*gx*nz*nv:4*nvar*gx*nz*nv] )    // THINK: why fail !!
                #pragma acc update device( pbX[0:4*npbX] )    // OK
                }
                #pragma omp section
                {
                #pragma acc update self( pbZ[0:2*npbZ] )
                MPI_Win_fence(0, w_pbZ);
                MPI_Put(&pbZ[0],    1, t_pbZ, lowerZ, 2 /* skip old Z block */, 1, t_pbZ, w_pbZ);
                MPI_Put(&pbZ[npbZ], 1, t_pbZ, upperZ, 3 /* skip old Z block */, 1, t_pbZ, w_pbZ);
                MPI_Win_fence(0, w_pbZ);
                #pragma acc update device( pbZ[0*npbZ:4*npbZ] )
                }
            }

            #if 0
            cout << "== AfterX: ";
            for (int i=0;i<5;i++) {
              cout << pbX[i+2*npbX] << " ";
            } cout << "\t Z: ";
            for (int i=0;i<5;i++) {
              cout << pbZ[i+2*npbZ] << " ";
            } cout << endl;
            #endif
#endif
        }

        void sync_buffer_isend() {
#ifdef COSENU_MPI
#if 0
            for (int i=0;i<5;i++) {
              pbX[i] = i;     pbX[i+2*nvar*gx*nz*nv] = 0;
              pbZ[i] = i;     pbZ[i+2*nvar*nx*gz*nv] = 0;
            }
            cout << "== BeforeX: ";
            for (int i=0;i<5;i++) {
              cout << pbX[i+2*nvar*gx*nz*nv] << " ";
            } cout << "\t Z: ";
            for (int i=0;i<5;i++) {
              cout << pbZ[i+2*nvar*nx*gz*nv] << " ";
            } cout << endl;
#endif

            MPI_Request reqs[8];
            MPI_Comm comm = CartCOMM; //MPI_COMM_WORLD;
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                const auto npbX = nvar*gx*nz*nv;
                MPI_Isend(&pbX[     0], 1, t_pbX, lowerX, 9, comm, &reqs[0]);
                MPI_Isend(&pbX[  npbX], 1, t_pbX, upperX, 9, comm, &reqs[1]);
                MPI_Irecv(&pbX[2*npbX], 1, t_pbX, upperX, 9, comm, &reqs[2]);
                MPI_Irecv(&pbX[3*npbX], 1, t_pbX, lowerX, 9, comm, &reqs[3]);
                }
                #pragma omp section
                {
                const auto npbZ = nvar*nx*gz*nv;
                MPI_Isend(&pbZ[     0], 1, t_pbZ, lowerZ, 9, comm, &reqs[4]);
    	        MPI_Isend(&pbZ[  npbZ], 1, t_pbZ, upperZ, 9, comm, &reqs[5]);
    	        MPI_Irecv(&pbZ[2*npbZ], 1, t_pbZ, upperZ, 9, comm, &reqs[6]);
    	        MPI_Irecv(&pbZ[3*npbZ], 1, t_pbZ, lowerZ, 9, comm, &reqs[7]);
                }
            }

            MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
#if 0
            cout << "== AfterX: ";
            for (int i=0;i<5;i++) {
              cout << pbX[i+2*nvar*gx*nz*nv] << " ";
            } cout << "\t Z: ";
            for (int i=0;i<5;i++) {
              cout << pbZ[i+2*nvar*nx*gz*nv] << " ";
            } cout << endl;
#endif
#endif
        }

        void sync_buffer_copy() {   // Only for single-node test
            memcpy(&pbX[2*gx*nz*nv*nvar], &pbX[0], 2*gx*nz*nv*nvar*sizeof(real));
            memcpy(&pbZ[2*nx*gz*nv*nvar], &pbZ[0], 2*nx*gz*nv*nvar*sizeof(real));
        }

        int  get_nv()    const {  return nv;   }
        int  get_nphi()  const  {  return nphi;   }
        ulong get_lpts()  const  {  return lpts;  }
        uint  get_lpts_x()  const  {  return lpts_x;  }
        uint  get_lpts_z()  const  {  return lpts_z;  }

        inline ulong idx(const int i, const int j, const int v) const { return ( (i+gx)*(nz+2*gz) + j+gz)*nv + v; }    //  i:x j:z

};
