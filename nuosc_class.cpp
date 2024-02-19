#include "nuosc_class.h"
#include "utils.h"

#if defined(IM_V2D_POLAR_GL_Z)
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
    #pragma omp parallel for simd collapse(2)
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
#endif

int gen_v2d_rsum_zphi(const int nv, const int nphi, Vec& vw, Vec &vx, Vec& vy, Vec& vz) {
    vx.reserve(nv*nphi);
    vy.reserve(nv*nphi);
    vz.reserve(nv*nphi);
    vw.reserve(nv*nphi);
    real dp = 2*M_PI/nphi;
    real dv = 2.0/(nv);     assert(nv%2==0);
    #pragma omp parallel for simd collapse(2)
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

NuOsc::NuOsc(int px_[], int nv_, const int nphi_, const int gx_[],
             const real bbox_[][2], const real dx_, const real CFL_, const real ko_) :
                 phy_time(0.), ko(ko_), dx(dx_), nphi(nphi_) {

    ds_L = dx*dx*dx/(bbox_[0][1]-bbox_[0][0])/(bbox_[1][1]-bbox_[1][0])/(bbox_[2][1]-bbox_[2][0]);
    CFL = CFL_;
    dt = dx*CFL;

    #ifdef COSENU_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //
    // Create Cartesian topology
    //
    int period[3] = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD, DIM, px_, period, 0, &CartCOMM);
    for (int d=0;d<DIM;++d) px[d] = px_[d];   // MPI will determine new px if px_={0,0,0} 

    // calcute local geometry from computational domain (x0_, x1_, z0_, z1_)
    MPI_Cart_coords(CartCOMM, myrank, DIM, rx);
    for (int d=0;d<DIM;++d) MPI_Cart_shift(CartCOMM, d, 1, &nb[d][0], &nb[d][1]);

    // Get shared commnicator, which is used to determine the ranks within a node (shared memory)
    MPI_Comm scomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &scomm);
    MPI_Comm_rank(scomm, &srank);
    if (!myrank) printf("[%.4f] Cartesian MPI commincator done.\n", utils::msecs_since());
    #else
    for (int d=0;d<DIM;++d) px[d] = 1;
    #endif

    // Local bbox and coordinates
    for (int d=0;d<DIM;++d) {
        gx[d] = gx_[d];

        bbox[d][0] = bbox_[d][0] + rx[d]    *(bbox_[d][1]-bbox_[d][0])/px[d];
        bbox[d][1] = bbox_[d][0] + (rx[d]+1)*(bbox_[d][1]-bbox_[d][0])/px[d];
        if (1 == px[d] - rx[d]) bbox[d][1] = bbox_[d][1];     // the leftmost processor
        nx[d] = int((bbox[d][1]-bbox[d][0])/dx);

        X[d].reserve(nx[d]);
        for(int i=0;i<nx[d]; ++i) X[d][i] = bbox[d][0] + (i+0.5)*dx;
    }

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
    #elif defined(SYNC_MPI_SENDRECV)
    if (!myrank) printf("MPI Sendrecv.\n");
    #else
    if (!myrank) printf("MPI non-blocking send / recv.\n");
    #endif

    // Determine nv, which could be complicated for say ICOSA grid.
    #if defined(IM_V2D_POLAR_GL_Z)
    nv = gen_v2d_GL_zphi(nv_,nphi_, vw, vx, vy, vz);
    #else
    nv = gen_v2d_rsum_zphi(nv_,nphi_, vw, vx, vy, vz);
    #endif
    if (!myrank) printf("[%.4f] v-grid generation done\n", utils::msecs_since());

    ulong size = nv;
    for (int d=0;d<DIM;++d)  size *= (nx[d]+2*gx[d]);
    lpts = size;
    real mem_per_var = lpts*8/1024./1024./1024;

    //
    // Local geometry determinded.  (nx,ny,nz,nv)
    // Now, prepare datatype for ghostzone block of each dimension
    //
    ulong nXYZV = nvar*nv;
    for (int d=0;d<DIM;++d) nXYZV *= nx[d];

    #pragma acc enter data create(this)
    for (int d=0;d<DIM;++d) {
        const ulong npb = nXYZV/nx[d]*gx[d];   // total size of halo
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

    #ifdef DEBUG
    print_info();
    #endif

    omp_set_max_active_levels(3);  // mainly for calRHS.

    if (myrank==0) {
            printf("\nNuOsc on %d (%dx%dx%d) MPI ranks: %d core per rank.\n", ranks, px[0], px[1], px[2], omp_get_max_threads() );
            printf("   Domain:  v: nv = %5d  ( w/ nphi = %5d ) on S2.\n", get_nv(), get_nphi() );
            printf("            x:( %12f %12f )  dx = %g\n", bbox_[0][0], bbox_[0][1], dx);
            printf("            y:( %12f %12f )  dy = %g\n", bbox_[1][0], bbox_[1][1], dx);
            printf("            z:( %12f %12f )  dz = %g\n", bbox_[2][0], bbox_[2][1], dx);
            printf("   Local size per field var = %.2f GB, totol memory per rank roughly %.2f GB\n", mem_per_var, mem_per_var*50);
            printf("   dt = %g     CFL = %g\n", dt, CFL);
#ifdef BC_PERI
            printf("   Use Periodic boundary\n");
#else
            printf("   Use open boundary\n");
#endif

#if defined(IM_V2D_POLAR_GL_Z)
            printf("   Use Gauss-Lobatto z-grid and uniform phi-grid.\n");
#else
            printf("   Use uniform z- and phi- grid.\n");
#endif

#ifndef KO_ORD_3
            printf("   Use 5-th order KO dissipation, KO eps = %g\n", ko);
#else
            printf("   Use 3-th order KO dissipation, KO eps = %g\n", ko);
#endif

#ifndef ADVEC_OFF
            printf("   Advection ON. (Center-FD)\n");
            //printf("   Use upwinded for advaction. (EXP. Always blowup!!\n");
            //printf("   Use lopsided FD for advaction\n");
#else
            printf("   Advection OFF.\n");
#endif

#ifdef VACUUM_OFF
            printf("   Vacuum term OFF.\n");
#else
            printf("   Vacuum term ON:  pmo= %g  theta= %g.\n", pmo, theta);
#endif
        }

        // field variables for analysis~~
        G0  = new real[size];
        G0b = new real[size];
        P1  = new real[size];
        P2  = new real[size];
        P3  = new real[size];
        P1b = new real[size];
        P2b = new real[size];
        P3b = new real[size];
        dP  = new real[size];
        dN  = new real[size];
        dPb = new real[size];
        dNb = new real[size];
#pragma acc enter data create(G0[0:size],G0b[0:size],P1[0:size],P2[0:size],P3[0:size],P1b[0:size],P2b[0:size],P3b[0:size],dP[0:size],dN[0:size],dPb[0:size],dNb[0:size])

        // field variables~~
        v_stat = new FieldVar(size);
        v_rhs  = new FieldVar(size);
        v_pre  = new FieldVar(size);
        v_cor  = new FieldVar(size);
        v_stat0 = new FieldVar(size);
#pragma acc enter data create(v_stat[0:1], v_stat0[0:1], v_rhs[0:1], v_pre[0:1], v_cor[0:1]) attach(v_stat, v_rhs, v_pre, v_cor, v_stat0)

        if (myrank==0) {
            anafile.open("analysis.dat", std::ofstream::out | std::ofstream::trunc);
            if(!anafile) cout << "*** Open fails: " << "./analysis.dat" << endl;
            anafile << "### [ phy_time,   1:maxrelP,    2:surv, survb,    4:avgP, avgPb,      6:aM0    7:Lex   8:ELNe]" << endl;
        }

        {   // Hvac
#if NFLAVOR == 3
            const real dms12=7.39e-5*1267.*2., dms13=2.5229e-3* 1267.*2., theta12=33.82/180., theta13=8.61/180., theta23=48.3/180., theta_cp=0.;
            const real c12=cos(theta12), s12=sin(theta12), c13=cos(theta13), s13=sin(theta13), c23=cos(theta23), s23=sin(theta23);
            const real Ue2 = s12*c13;
            const real Um2r= c12*c23-s12*s13*s23*cos(theta_cp);
            const real Ut2r=-c12*s23-s12*s13*c23*cos(theta_cp);
            const real Um2i=        -s12*s13*s23*sin(theta_cp);
            const real Ut2i=        -s12*s13*c23*sin(theta_cp);
            const real Ue3r= s13*cos(theta_cp);
            const real Ue3i=-s13*sin(theta_cp);
            const real Um3 = c13*s23;
            const real Ut3 = c13*c23;

            hee  = dms12*(Ue2*Ue2)            +dms13*(s13*s13);
            hmm  = dms12*(Um2r*Um2r+Um2i*Um2i)+dms13*(Um3*Um3);
            htt  = dms12*(Ut2r*Ut2r+Ut2i*Ut2i)+dms13*(Ut3*Ut3);
            hemr = dms12*( Ue2*Um2r)          +dms13*(Ue3r*Um3);
            hemi = dms12*(-Ue2*Um2i)          +dms13*(Ue3i*Um3);
            hmtr = dms12*( Ue2*Ut2r)          +dms13*(Ue3r*Ut3);
            hmti = dms12*(-Ue2*Ut2i)          +dms13*(Ue3i*Ut3);
            hter = dms12*(Um2r*Ut2r+Um2i*Ut2i)+dms13*(Um3*Ut3);
            htei = dms12*(Ut2r*Um2i-Um2r*Ut2i);
            real Hvac_trace = (hee+hmm+htt)/3.0;
            hee -= Hvac_trace;
            hmm -= Hvac_trace;
            htt -= Hvac_trace;
#elif NFLAVOR == 2
            const real theta = 37 * M_PI / 180.;  //1e-6;
            const real ct = cos(2*theta);
            const real st = sin(2*theta);
#endif
        }
}
