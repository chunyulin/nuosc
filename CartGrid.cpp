#include "CartGrid.h"
#include "icosahedron.h"
//#include "jacobi_poly.h"

/*
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
        for (int i=0;i<nv;   ++i)   {
            vz[j*nv+i] = sqrt(1-r[i]*r[i]);
            vx[j*nv+i] = cos(j*dp)*vz[j*nv+i];
            vy[j*nv+i] = sin(j*dp)*vz[j*nv+i];
            vz[j*nv+i] = r[i];
            vw[j*nv+i] = w[i]/nphi;
        }
    return nv*nphi;
}
*/

int gen_v2d_rsum_zphi(const int nv, const int nphi, Vec& vw, Vec &vx, Vec& vy, Vec& vz) {
    vx.reserve(nv*nphi);
    vy.reserve(nv*nphi);
    vz.reserve(nv*nphi);
    vw.reserve(nv*nphi);
    real dp = 2*M_PI/nphi;
    real dv = 2.0/(nv);
    for (int j=0;j<nphi; ++j)
        for (int i=0;i<nv;   ++i)   {
            real tmp = (i+0.5)*dv - 1;
            vz[j*nv+i] = sqrt(1-tmp*tmp);
            vx[j*nv+i] = cos(j*dp)*vz[j*nv+i];
            vy[j*nv+i] = sin(j*dp)*vz[j*nv+i];
            vz[j*nv+i] = tmp;
            vw[j*nv+i] = dv/nphi;
        }
    return nv*nphi;
}

int gen_v2d_icosahedron(const int nv_, Vec& vw, Vec& vx, Vec& vy, Vec& vz) {
    
    IcosahedronVoronoi icosa(nv_);
    int nv=icosa.N;
    vx.reserve(nv);
    vy.reserve(nv);
    vz.reserve(nv);
    vw.reserve(nv);
    for (int i=0; i<nv; ++i) {
        vx[i] = icosa.X[i].x;
        vy[i] = icosa.X[i].y;
        vz[i] = icosa.X[i].z;
        vw[i] = icosa.vw[i];
    }
    return nv;
}

/*
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
}*/

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
CartGrid::CartGrid(int px_, int pz_, int nv_, int nphi_, int gx_, int gz_, real x0_, real x1_, real z0_, real z1_, real dx_, real dz_) 
: px(px_), pz(pz_), nphi(nphi_), gx(gx_), gz(gz_), dx(dx_), dz(dz_), nvar(8)
{
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
    
    //
    // Get shared comm
    //
    MPI_Comm scomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &scomm);
    MPI_Comm_rank(scomm, &srank);

    // set GPU
    #ifdef _OPENACC
    auto dev_type = acc_get_device_type();
    ngpus = acc_get_num_devices( dev_type );
    acc_set_device_num( srank, dev_type );
    #ifdef GDR_OFF
    if (!myrank) printf("\nOpenACC Enabled with %d GPU per node. (GDR = OFF)\n", ngpus );
    #else
    if (!myrank) printf("\nOpenACC Enabled with %d GPU per node. (GDR = ON)\n", ngpus );
    #endif
    #endif

    #ifdef SYNC_NCCL
    ncclUniqueId id;
    if (myrank == 0) ncclGetUniqueId(&id);
    #ifdef COSENU_MPI
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    #endif
    ncclCommInitRank(&_ncclcomm, ranks, id, myrank);     // should this be put after acc_set_device() ?

    for (int i=0;i<2*DIM;++i)  cudaStreamCreate(&stream[i]);
    #endif

    #if defined(SYNC_NCCL)
    if (!myrank) {
        printf("Sync by NCCL.\n");
    }
    #elif defined(SYNC_COPY)
    if (!myrank) printf("SYNC_COPY for test.\n");
    #elif defined(SYNC_MPI_ONESIDE_COPY)
    if (!myrank) printf("Sync by MPI One-side copy.\n");
    #elif defined(SYNC_MPI_SENDRECV)
    if (!myrank) printf("Sync by MPI Sendrecev.\n");
    #else
    if (!myrank) printf("Sync by MPI Isend / Irecv.\n");
    #endif


#endif

    x0 = x0_ + (x1_-x0_)*rx    /px;
    x1 = x0_ + (x1_-x0_)*(rx+1)/px;  if (rx+1 == px)  x1 = x1_;
    z0 = z0_ + (z1_-z0_)*rz    /pz;
    z1 = z0_ + (z1_-z0_)*(rz+1)/pz;  if (rz+1 == pz)  z1 = z1_;

    nx  = int((x1-x0)/dx);
    nz  = int((z1-z0)/dz);

    X.reserve(nx);   for(int i=0;i<nx; ++i)    X[i] = x0 + (i+0.5)*dx;
    Z.reserve(nz);   for(int i=0;i<nz; ++i)    Z[i] = z0 + (i+0.5)*dz;

#if defined(IM_V2D_POLAR_GRID)
    nv = gen_v2d_rsum_zphi(nv_,nphi_, vw, vx, vy, vz);
#else
    nv = gen_v2d_icosahedron(nv_, vw, vx, vy, vz);
#endif

    lpts = (nx+2*gx)*(nz+2*gz)*nv;
    //lpts_x = gx*nz
    //lpts_z = gz*nx

    // prepare datatype for ghostzone block of each dimension
    const auto npbX = nvar*gx*nz*nv;
    const auto npbZ = nvar*nx*gz*nv;

#ifdef COSENU_MPI
    MPI_Type_contiguous(npbX, MPI_DOUBLE, &t_pbX);  MPI_Type_commit(&t_pbX);
    MPI_Type_contiguous(npbZ, MPI_DOUBLE, &t_pbZ);  MPI_Type_commit(&t_pbZ);

    #ifdef SYNC_MPI_ONESIDE_COPY
    // prepare (un-)pack buffer and MPI RMA window for sync. (duplicate 4 times for left/right and old/new)
    int ierr = 0;
    ierr = MPI_Win_allocate(4*npbX*sizeof(real), npbX*sizeof(real), MPI_INFO_NULL, CartCOMM, &pbX, &w_pbX);
    ierr = MPI_Win_allocate(4*npbZ*sizeof(real), npbZ*sizeof(real), MPI_INFO_NULL, CartCOMM, &pbZ, &w_pbZ);
    if (ierr!=MPI_SUCCESS) {  cout << " !! MPI error " << ierr << " at " << __FILE__ << ":" << __LINE__ << endl; }
    #else
    pbX = new real[4*npbX];
    pbZ = new real[4*npbZ];
    #pragma acc enter data create(this,pbX[0:4*npbX],pbZ[0:4*npbZ])
    #endif

#else
    pbX = new real[4*npbX];
    pbZ = new real[4*npbZ];
    #pragma acc enter data create(this,pbX[0:4*npbX],pbZ[0:4*npbZ])
#endif

    for (int i=0;i<ranks;++i) {
       if (myrank==i)  {
          print_info();
       }
       #ifdef COSENU_MPI
       MPI_Barrier(MPI_COMM_WORLD);
       #endif
    }
    //test_win();
}

CartGrid::~CartGrid() {
    //MPI_Win_free(&w_pbX);   // will free by the MPI_Finalize() anyway.
    //MPI_Win_free(&w_pbZ);
    //MPI_Type_free(&t_pbX);
    //MPI_Type_free(&t_pbZ);

//#pragma acc exit data delete(pbX,pbZ)

#ifndef COSENU_MPI
    delete[] pbX;
    delete[] pbZ;
#endif
}


void CartGrid::sync_put() {
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
        #ifdef GDR_OFF
            #pragma acc update self( pbX[0:2*npbX] )
        #else
            #pragma acc host_data use_device(pbX)
        #endif
        {
            MPI_Win_fence(0, w_pbX);
            MPI_Put(&pbX[0],    1, t_pbX, lowerX, 2 /* skip old X block */, 1, t_pbX, w_pbX);
            MPI_Put(&pbX[npbX], 1, t_pbX, upperX, 3 /* skip old X block */, 1, t_pbX, w_pbX);
            MPI_Win_fence(0, w_pbX);
        }
        #ifdef GDR_OFF
        #pragma acc update device( pbX[2*npbX:4*npbX] )
        #endif
#pragma omp section
        #ifdef GDR_OFF
            #pragma acc update self( pbZ[0:2*npbZ] )
        #else
            #pragma acc host_data use_device(pbZ)
        #endif
        {
            MPI_Win_fence(0, w_pbZ);
            MPI_Put(&pbZ[0],    1, t_pbZ, lowerZ, 2 /* skip old Z block */, 1, t_pbZ, w_pbZ);
            MPI_Put(&pbZ[npbZ], 1, t_pbZ, upperZ, 3 /* skip old Z block */, 1, t_pbZ, w_pbZ);
            MPI_Win_fence(0, w_pbZ);
        }
        #ifdef GDR_OFF
        #pragma acc update device( pbZ[2*npbZ:4*npbZ] )
        #endif
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

void CartGrid::sync_isend() {
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
    const auto npbX = nvar*gx*nz*nv;
    const auto npbZ = nvar*nx*gz*nv;
#pragma omp parallel sections
    {
#pragma omp section
        #ifdef GDR_OFF
          #pragma acc update self( pbX[0:2*npbX] )
        #else
          #pragma acc host_data use_device(pbX)
        #endif
        {
            MPI_Isend(&pbX[     0], 1, t_pbX, lowerX,  9, comm, &reqs[0]);
            MPI_Isend(&pbX[  npbX], 1, t_pbX, upperX, 10, comm, &reqs[1]);
            MPI_Irecv(&pbX[2*npbX], 1, t_pbX, upperX,  9, comm, &reqs[2]);
            MPI_Irecv(&pbX[3*npbX], 1, t_pbX, lowerX, 10, comm, &reqs[3]);
        }
#pragma omp section
        #ifdef GDR_OFF
          #pragma acc update self( pbZ[0:2*npbZ] )
        #else
          #pragma acc host_data use_device(pbZ)
        #endif
        {
            MPI_Isend(&pbZ[     0], 1, t_pbZ, lowerZ, 19, comm, &reqs[4]);
            MPI_Isend(&pbZ[  npbZ], 1, t_pbZ, upperZ, 20, comm, &reqs[5]);
            MPI_Irecv(&pbZ[2*npbZ], 1, t_pbZ, upperZ, 19, comm, &reqs[6]);
            MPI_Irecv(&pbZ[3*npbZ], 1, t_pbZ, lowerZ, 20, comm, &reqs[7]);
        }
    }

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
    #ifdef GDR_OFF
    #pragma acc update device( pbX[(2*npbX):(4*npbX)], pbZ[(2*npbZ):(4*npbZ)] )
    #endif
    
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

void CartGrid::sync_copy() {   // Only for single-node test
    memcpy(&pbX[2*gx*nz*nv*nvar], &pbX[0], 2*gx*nz*nv*nvar*sizeof(real));
    memcpy(&pbZ[2*nx*gz*nv*nvar], &pbZ[0], 2*nx*gz*nv*nvar*sizeof(real));
}

// FIXME: SendRecv gives wrong result under GPU mode even for Rank=1 !!
void CartGrid::sync_sendrecv() {
    #ifdef COSENU_MPI
    assert(0);
    #endif
}

#ifdef SYNC_NCCL
void CartGrid::sync_nccl() {
    #ifdef NVTX
    nvtxRangePush("Sync NCCL");
    #endif
    const auto npbX = nvar*gx*nz*nv;
    const auto npbZ = nvar*nx*gz*nv;

    NCCLCHECK( ncclGroupStart() );
        NCCLCHECK( ncclSend(&pbX[     0], npbX, ncclDouble, lowerX, _ncclcomm, stream[0]) );
        NCCLCHECK( ncclRecv(&pbX[2*npbX], npbX, ncclDouble, upperX, _ncclcomm, stream[0]) );
    NCCLCHECK( ncclGroupEnd() );
    
    NCCLCHECK( ncclGroupStart() );
        NCCLCHECK( ncclSend(&pbX[  npbX], npbX, ncclDouble, upperX, _ncclcomm, stream[1]) );
        NCCLCHECK( ncclRecv(&pbX[3*npbX], npbX, ncclDouble, lowerX, _ncclcomm, stream[1]) );
    NCCLCHECK( ncclGroupEnd() );

    NCCLCHECK( ncclGroupStart() );
        NCCLCHECK( ncclSend(&pbZ[     0], npbZ, ncclDouble, lowerZ, _ncclcomm, stream[2]) );
        NCCLCHECK( ncclRecv(&pbZ[2*npbZ], npbZ, ncclDouble, upperZ, _ncclcomm, stream[2]) );
    NCCLCHECK( ncclGroupEnd() );
    
    NCCLCHECK( ncclGroupStart() );
        NCCLCHECK( ncclSend(&pbZ[  npbZ], npbZ, ncclDouble, upperZ, _ncclcomm, stream[3]) );
        NCCLCHECK( ncclRecv(&pbZ[3*npbZ], npbZ, ncclDouble, lowerZ, _ncclcomm, stream[3]) );
    NCCLCHECK( ncclGroupEnd() );

    cudaDeviceSynchronize();

    #ifdef NVTX
    nvtxRangePop();
    #endif
}
#endif


