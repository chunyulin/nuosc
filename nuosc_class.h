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

#define COLLAPSE_LOOP 4
#define PARFORALL(i,j,k,v) \
    _Pragma("acc parallel loop collapse(4)") \
    _Pragma("omp parallel for collapse(4)") \
    for (int i=0;i<nx[0]; ++i) \
    for (int j=0;j<nx[1]; ++j) \
    for (int k=0;k<nx[2]; ++k) \
    for (int v=0;v<nv; ++v)

#define FORALL(i,j,k,v) \
    for (int i=0;i<nx[0]; ++i) \
    for (int j=0;j<nx[1]; ++j) \
    for (int k=0;k<nx[2]; ++k) \
    for (int v=0;v<nv; ++v)

enum ff {
     ee = 0,  mm,  emr,  emi,
    bee,     bmm, bemr, bemi,     // Total  8 flavor fields.
#if NFLAVOR == 3
     tt,  mtr,  mti,  ter,   tei,
    btt, bmtr, bmti, bter,  btei, // Total 18 flavor fields.
#endif
};

struct FieldVar {

#if 1
    std::array<std::vector<real>, 2*NFLAVOR*NFLAVOR> wf;

    FieldVar(int size) {
        for (int f=0;f<2*NFLAVOR*NFLAVOR; ++f)
            wf[f] = std::vector<real>(size,0); // initialize to zero
    }
#else
    std::array<real*, 2*NFLAVOR*NFLAVOR> wf;
    FieldVar(int size) {
        for (int f=0;f<2*NFLAVOR*NFLAVOR; ++f)  wf[f] = new real[size]();
    }
    ~FieldVar() {
        for (int f=0;f<2*NFLAVOR*NFLAVOR; ++f)  delete[] wf[f];
    }

#endif
};
#ifdef WENO7
struct Flux {
    int size;
    real *l2h; // Flux: from low to high.
    real *h2l; // Flux: from high to low.
    Flux(int size_) : size(size_)  {
        l2h = new real[size]();
        h2l = new real[size]();
    }
    ~Flux() {
        delete[] l2h;
        delete[] h2l;
    }
};
#endif

typedef struct SnapShot_struct {
    std::list<std::vector<real>> var_list;
    string fntpl;
    int every;
    std::vector<int> x_slices;   // coordinate for the reduced dimension
    std::vector<int> v_slices;

    // init with specified v-coordinate
    SnapShot_struct(std::list<std::vector<real>> var_list_, string fntpl_, int every_,  std::vector<int> v_slices_) {
        var_list = var_list_;
        fntpl = fntpl_;
        every = every_;
        v_slices = v_slices_;
    }
    // init with specified y and v-coordinate
    SnapShot_struct(std::list<std::vector<real>> var_list_, string fntpl_, int every_, std::vector<int> x_slices_, std::vector<int> v_slices_) {
        var_list = var_list_;
        fntpl = fntpl_;
        every = every_;
        x_slices = x_slices_;
        v_slices = v_slices_;
    }
} SnapShot;

inline void swap(FieldVar **a, FieldVar **b) { FieldVar *tmp = *a; *a = *b; *b = tmp; }
inline real random_amp(real a) { return a * rand() / RAND_MAX; }
template <typename T> int sgn(T val) {    return (T(0) < val) - (val < T(0));   }

std::vector<int> gen_skimmed_vslice_index(int nv_target, int nv_in);

class NuOsc {
    public:
        const int nvar = 2*NFLAVOR*NFLAVOR;
        int ranks = 1, myrank = 0;

        real phy_time;
        real dt, dx;       // dx, dy, dz
        real ds_L;         // = dx*dy*dz/(z1-z0)/(y1-y0)/(x1-x0)

        int  px[DIM];       // processor geometry: px, py, pz
        int  nx[DIM];       // local shape: nx, ny, nz
        int  gx[DIM];       // ghost zone width
        Vec  X[DIM];        // coordinates X,Y,Z
        real bbox[DIM][2];  // bounding box [x0,x1] [y0,y1] [z0,z1]

        int nv, nphi;       // # of v cubature points.
        Vec vz, vx, vy, vw; // v coors and integral quadrature

        ulong lpts;

        int srank = 0;
        int rx[DIM] = {0};       // Index of my processor
        int nb[DIM][2] = {0};    // Cartensian neighbor ranks

        int ngpus = 0;
#ifdef COSENU_MPI
        MPI_Comm CartCOMM;
        // packing buffers for X- X+, Y- Y+, and Z- Z+.
        MPI_Datatype t_pb[DIM];
        MPI_Win      w_pb[DIM];

        MPI_Request reqs[DIM*4];
#endif
        real *pb[DIM];     // TODO: can we use the storage of FieldVar directly w/o copying to this buffers ?

//--------------

        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;  // field variables
        FieldVar *v_stat0;   // NOT used.
#ifdef WENO7
        Flux *flux;
#endif
        real *P1,  *P2,  *P3,  *dN,  *dP;
        real *P1b, *P2b, *P3b, *dNb, *dPb;
        real *G0,*G0b;
        real n_nue0[2];   // initial number density for nue/nueb

        real CFL;
        real ko;

        const real theta = 37 * M_PI / 180.;  //1e-6;
#if NFLAVOR == 3
        real hee,hmm,htt,hemr,hemi,hmtr,hmti,hter,htei;
#elif NFLAVOR == 2
        const real ct = cos(2*theta);
        const real st = sin(2*theta);
#else
#error NFLAVOR can be 2 or 3 only.
#endif
        
        real pmo = 0.1;      // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
        real mu  = 1.0;      // can be set by set_mu()
        bool renorm = false;  // can be set by set_renorm()

        std::ofstream anafile;
        std::list<SnapShot> snapshots;

        #ifdef PROFILING_BREAKDOWNS
        float t_step=0, t_sync=0, t_packing=0;
        #endif

        NuOsc(int px_[], int nv_, const int nphi_, const int gx_[],
              const real bbox[][2], const real dx_, const real CFL_, const real  ko_);

        ~NuOsc() {
            //#pragma acc exit data delete(G0,G0b,P1,P2,P3,P1b,P2b,P3b,dP,dN,dPb,dNb)
            delete[] G0;
            delete[] G0b;
            delete[] P1;  delete[] P2;  delete[] P3;  delete[] dP;  delete[] dN;
            delete[] P1b; delete[] P2b; delete[] P3b; delete[] dPb; delete[] dNb;
            //#pragma acc exit data delete(v_stat, v_rhs, v_pre, v_cor, v_stat0)
            delete v_stat;  delete v_rhs; delete v_pre; delete v_cor; delete v_stat0;
            #ifdef WENO7
            delete flux;
            #endif
            anafile.close();

            for (int d=0;d<DIM;++d) {
               #if defined(COSENU_MPI)
               // Segfault if trying to deallocate any MPI object manully. TODO: put MPI_Init into main class instead of at main.
               //MPI_Type_free(&t_pb[d]);   
               #if defined(SYNC_MPI_ONESIDE_COPY)
               //MPI_Win_free(&w_pb[d]);
               #endif
               #endif
            }
            #if ! ( defined(COSENU_MPI) && defined(SYNC_MPI_ONESIDE_COPY) )
            for (int d=0;d<DIM;++d) delete[] pb[d];
            #endif
        }

        void set_mu(real mu_) {
            mu = mu_;
            if (myrank==0) printf("   Setting mu = %f\n", mu);
        }
        void set_pmo(real pmo_) {
            pmo = pmo_;
            if (myrank==0) printf("   Setting pmo = %f\n", pmo);
        }
        void set_renorm(bool renorm_) {
            renorm = renorm_;
            if (myrank==0) printf("   Setting renorm = %d\n", renorm);
        }
        ulong get_lpts() const {  return lpts;  }
        int  get_nv() const {  return nv;  }
        int  get_nphi()  const  {  return nphi;   }


        void fillInitValue(int ipt, real alpha, real eps0, real sigma, real lnue, real lnueb, real lnuex, real lnuebx);
        void fillInitGaussian(real eps0, real sigma);
        void fillInitSquare(real eps0, real sigma);
        void fillInitTriangle(real eps0, real sigma);
        void updatePeriodicBoundary (FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);
        void step_rk4();
        void calRHS(FieldVar* out, FieldVar * in);
        //void calRHS_core(FieldVar* out, const FieldVar * in, const int bb[2*DIM]);
        void calRHS_with_bdry(FieldVar* out, const FieldVar * in);
        void calRHS_wo_bdry(FieldVar* out, const FieldVar * in);
        void packSend(const FieldVar* in);
        void pack_buffer(const FieldVar* in);
        void waitall();
        void unpack_buffer(FieldVar* v0);
        void sync_sendrecv();
        void sync_nonblocking();
        void sync_put();
        void sync_copy();
#ifdef WENO7
        void get_flux(Flux *, const std::vector<real>, const int, const int, const int, const int);
#endif

        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
        void analysis();
        void eval_conserved(const FieldVar* v0);
        void renormalize(FieldVar* v0);

        // 1D output:
        void addSnapShotAtV(std::list<std::vector<real>> var, char *fntpl, int dumpstep, std::vector<int>  vidx);
        void checkSnapShot(const int t=0) const;
        // 2D output:
        void addSnapShotAtXV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> xidx, std::vector<int> vidx);

//--------------
        inline void print_info() {
            char hname[20];
            char tag[50];
            if(0!=gethostname(hname,sizeof(hname))) { cout << "Error in gethostname" << endl; }

            if (ngpus > 0) sprintf(tag, "GPU %d %s", srank%ngpus, hname);
            else           sprintf(tag, "%s", hname);

            printf("[ Rank %2d ( %d %d %d ) on %s ]  nber:( %d %d )( %d %d )( %d %d )  N:[ %d %d %d %d ](nphi:%d)   bbox:( %g %g )( %g %g )( %g %g ) ]\n",
                   myrank, rx[0],rx[1],rx[2], tag, nb[0][0],nb[0][1],nb[1][0],nb[1][1],nb[2][0],nb[2][1], 
                   nx[0],nx[1],nx[2], nv,nphi, bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1]);
        }

#if DIM == 2
        inline ulong idx(const int i, const int j, const int k, const int v) const { return v + nv * ( (k+gx[2]) + (nx[1]+2*gx[1])*(  j+gx[1] )  ); }
#elif DIM == 3
        inline ulong idx(const int i, const int j, const int k, const int v) const { return v + nv * ( (k+gx[2]) + (nx[2]+2*gx[2])*( (j+gx[1])+ (nx[1]+2*gx[1])*(i+gx[0]) ) ); }
#endif

};

