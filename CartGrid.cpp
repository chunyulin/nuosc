#include "CartGrid.h"
#include "jacobi_poly.h"

int gen_v2d_GL_zphi(const int nv, const int nphi, Vec& vw, Vec& vx, Vec& vy, Vec& vz) {
    Vec r(nv);
    Vec w(nv);
    JacobiGL(nv-1,0,0,r,w);
    vx.reserve(nv*nphi);
    vy.reserve(nv*nphi);
    vz.reserve(nv*nphi);
    vw.reserve(nv*nphi);
    real dp = 2*M_PI/nphi;
    for (int j=0;j<nphi; ++j)
        for (int i=0;i<nv; ++i)   {
            real vxy = sqrt(1-r[i]*r[i]);
            vx[j*nv+i] = cos(j*dp)*vxy;
            vy[j*nv+i] = sin(j*dp)*vxy;
            vz[j*nv+i] = r[i];
            vw[j*nv+i] = w[i]/nphi;
        }
    return nv*nphi;
}

int gen_v2d_rsum_zphi(const int nv, const int nphi, Vec& vw, Vec &vx, Vec& vy, Vec& vz) {
    vx.reserve(nv*nphi);
    vy.reserve(nv*nphi);
    vz.reserve(nv*nphi);
    vw.reserve(nv*nphi);
    real dp = 2*M_PI/nphi;
    real dv = 2.0/(nv);     assert(nv%2==0);
    for (int j=0;j<nphi; ++j)
        for (int i=0;i<nv;   ++i)   {
            real tmp = (i+0.5)*dv - 1;
            real vxy = sqrt(1-tmp*tmp);
            vx[j*nv+i] = cos(j*dp)*vxy;
            vy[j*nv+i] = sin(j*dp)*vxy;
            vz[j*nv+i] = tmp;
            vw[j*nv+i] = dv/nphi;
        }
    return nv*nphi;
}

int gen_v1d_GL(const int nv, Vec vw, Vec vz) {
    Vec r(nv,0);
    Vec w(nv,0);
    JacobiGL(nv-1,0,0,r,w);
    vz.reserve(nv);
    vw.reserve(nv);
    for (int j=0;j<nv; ++j) {
        vz[j] = r[j];
        vw[j] = w[j];
    }
    return nv;
}

// v quaduture in [-1:1], vertex-center with simple trapezoidal rules.
int gen_v1d_trapezoidal(const int nv, Vec vw, Vec vz) {
    assert(nv%2==1);
    real dv = 2.0/(nv-1);
    vz.reserve(nv);
    vw.reserve(nv);
    for (int j=0;j<nv; ++j) {
        vz[j] = j*dv - 1;
        vw[j] = dv;
    }
    vw[0]    = 0.5*dv;
    vw[nv-1] = 0.5*dv;
    return nv;
}

// v quaduture in [-1:1], vertex-center with Simpson 1/3 rule on uniform rgid.
int gen_v1d_simpson(const int nv, Vec vw, Vec vz) {
    assert(nv%2==1);
    real dv = 2.0/(nv-1);
    vz.reserve(nv);
    vw.reserve(nv);
    const real o3dv = 1./3.*dv;
    for (int j=0;j<nv; j++) {
        vz[j] = j*dv - 1;
        vw[j] = 2*((j%2)+1)*o3dv;
    }
    vw[0]    = o3dv;
    vw[nv-1] = o3dv;
    return nv;
}

int gen_v1d_cellcenter(const int nv, Vec vw, Vec vz) {
    assert(nv%2==0);
    real dv = 2.0/(nv);
    vz.reserve(nv);
    vw.reserve(nv);
    for (int j=0;j<nv; ++j) {
        vz[j] = (j+0.5)*dv - 1;
        vw[j] = dv;
    }
    return nv;
}

// calculates local box and cartesian communicator given global box and processor shape
CartGrid::CartGrid(int px_[], int nv_, const int nphi_, const int gx_[], const real bbox_[][2], const real dx_)
: nphi(nphi_), nvar(8), dx(dx_)
{

#ifdef COSENU_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // create Cartesian topology
    int period[3] = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD, DIM, px_, period, 0, &CartCOMM);
    for (int d=0;d<DIM;++d) px[d] = px_[d];   // MPI will determine new px if px_={0,0,0} 

    // calcute local geometry from computational domain (x0_, x1_, z0_, z1_)
    MPI_Cart_coords(CartCOMM, myrank, DIM, rx);
    for (int d=0;d<DIM;++d) MPI_Cart_shift(CartCOMM, d, 1, &nb[d][0], &nb[d][1]);

    //
    // Get shared comm
    //
    MPI_Comm scomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &scomm);
    MPI_Comm_rank(scomm, &srank);
#else
    for (int d=0;d<DIM;++d) px[d] = 1;
#endif

    // set GPU
    #ifdef _OPENACC
    auto dev_type = acc_get_device_type();
    ngpus = acc_get_num_devices( dev_type );
    acc_set_device_num( srank%ngpus, dev_type );
    #ifdef GDR_OFF
    if (!myrank) printf("\nOpenACC Enabled with %d GPU per node. (GDR = OFF)\n", ngpus );
    #else
    if (!myrank) printf("\nOpenACC Enabled with %d GPU per node. (GDR = ON)\n", ngpus );
    #endif
    #endif

    #if defined(SYNC_COPY)
    if (!myrank) printf("SYNC_COPY for test.\n");
    #elif defined(SYNC_MPI_ONESIDE_COPY)
    if (!myrank) printf("MPI One-side copy.\n");
    #elif defined(SYNC_MPI_ISEND)
    if (!myrank) printf("MPI Isendrecv.\n");
    #else
    if (!myrank) printf("MPI Isend / Irecv.\n");
    #endif
    
    // Local bbox and coordinates
    for (int d=0;d<DIM;++d) {
        gx[d] = gx_[d];

        bbox[d][0] = bbox_[d][0] + rx[d]    *(bbox_[d][1]-bbox_[d][0])/px[d];
        bbox[d][1] = bbox_[d][0] + (rx[d]+1)*(bbox_[d][1]-bbox_[d][0])/px[d];
        if (1 == px[d] - rx[d]) bbox[d][1] = bbox_[d][1];
        nx[d] = int((bbox[d][1]-bbox[d][0])/dx);
 
        X[d].reserve(nx[d]);
        for(int i=0;i<nx[d]; ++i) X[d][i] = bbox[d][0] + (i+0.5)*dx;
    }

#if defined(IM_V2D_POLAR_GL_Z)
    nv = gen_v2d_GL_zphi(nv_,nphi_, vw, vx, vy, vz);
#else
    nv = gen_v2d_rsum_zphi(nv_,nphi_, vw, vx, vy, vz);
#endif

    lpts = nv;
    for (int d=0;d<DIM;++d)  lpts *= (nx[d]+2*gx[d]);

    // prepare datatype for ghostzone block of each dimension
    int nXYZV = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZV *= nx[d];

    #pragma acc enter data create(this)
    for (int d=0;d<DIM;++d) {
        const auto npb = nXYZV/nx[d]*gx[d];   // total size of halo
    #ifdef COSENU_MPI
        MPI_Type_contiguous(npb, MPI_DOUBLE, &t_pb[d]);  MPI_Type_commit(&t_pb[d]);

        #ifdef SYNC_MPI_ONESIDE_COPY
        // prepare (un-)pack buffer and MPI RMA window for sync. (duplicate 4 times for left/right and old/new)
        int ierr = 0;
        ierr = MPI_Win_allocate(4*npb*sizeof(real), npb*sizeof(real), MPI_INFO_NULL, CartCOMM, &pb[d], &w_pb[d]);
        if (ierr!=0) { cout << "MPI_Win_allocate error!" << endl; exit(0); }
        #else
        pb[d] = new real[4*npb];
        #endif

    #else
        pb[d] = new real[4*npb];
    #endif

        #pragma acc enter data create(pb[d][0:4*npb])
    }

#ifdef COSENU_MPI
    for (int i=0;i<ranks;++i) {
       if (myrank==i)  {
          print_info();
          MPI_Barrier(MPI_COMM_WORLD);
       } else {
          MPI_Barrier(MPI_COMM_WORLD);
       }
    }
#else
    print_info();
#endif
    //test_win();
}

CartGrid::~CartGrid() {
    //MPI_Win_free(&w_pbX);   // will free by the MPI_Finalize() anyway.
    //MPI_Win_free(&w_pbZ);
    //MPI_Type_free(&t_pbX);
    //MPI_Type_free(&t_pbZ);
    for (int d=0;d<DIM;++d) {
       #pragma acc exit data delete(pb[d])
    }

#ifndef COSENU_MPI
    for (int d=0;d<DIM;++d) delete[] pb[d];
#endif
}


void CartGrid::sync_put() {
#ifdef NVTX
    nvtxRangePush("Sync Put");
#endif
#ifdef COSENU_MPI
    int nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];
        
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const auto npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #ifdef GDR_OFF
            #pragma acc update self( pb[d][0:2*npb] )
        #else
            #pragma acc host_data use_device(pb[d])
        #endif
        {
            MPI_Win_fence(0, w_pb[d]);
            MPI_Put(&pb[d][0],   1, t_pb[d], nb[0][0], 2 /* skip old block */, 1, t_pb[d], w_pb[d]);
            MPI_Put(&pb[d][npb], 1, t_pb[d], nb[0][1], 3 /* skip old block */, 1, t_pb[d], w_pb[d]);
            MPI_Win_fence(0, w_pb[d]);
        }
        #ifdef GDR_OFF
        #pragma acc update device( pb[d][0:4*npb] )
        #endif
    }
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void CartGrid::sync_isend() {
#ifdef NVTX
    nvtxRangePush("Sync Isend");
#endif
#ifdef COSENU_MPI
    int nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    const int n_reqs = DIM*4;
    MPI_Request reqs[n_reqs];
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const auto npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #ifdef GDR_OFF
          #pragma acc update self( pb[d][0:2*npb] )
        #else
          #pragma acc host_data use_device(pb[d])
        #endif
        {
            MPI_Irecv(&pb[d][2*npb], 1, t_pb[d], nb[d][1],   d*2, CartCOMM, &reqs[d*4]);
            MPI_Irecv(&pb[d][3*npb], 1, t_pb[d], nb[d][0], 1+d*2, CartCOMM, &reqs[d*4+1]);
            MPI_Isend(&pb[d][    0], 1, t_pb[d], nb[d][0],   d*2, CartCOMM, &reqs[d*4+2]);
            MPI_Isend(&pb[d][  npb], 1, t_pb[d], nb[d][1], 1+d*2, CartCOMM, &reqs[d*4+3]);
        }
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    #ifdef GDR_OFF
    #pragma omp parallel for num_threads(DIM)
    for (int d=0;d<DIM;++d) {
        const auto npb = nXYZ/nx[d]*gx[d];   // total size of halo
        #pragma acc update device( pb[d][(2*npb):(4*npb)] )
    }
    #endif
#endif
#ifdef NVTX
    nvtxRangePop();
#endif
}

void CartGrid::sync_copy() {   // Only for single-node test

    int nXYZ = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZ *= nx[d];

    for (int d=0;d<DIM;++d) {
        const auto npb = nXYZ/nx[d]*gx[d];   // total size of halo
        memcpy(&pb[d][2*npb], &pb[d][0], 2*npb*sizeof(real));
    }
}

