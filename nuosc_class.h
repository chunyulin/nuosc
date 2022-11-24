#pragma once
#include "common.h"
#include "CartGrid.h"

#define COLLAPSE_LOOP 4
#define PARFORALL(i,j,k,v) \
    _Pragma("omp parallel for collapse(4)") \
    _Pragma("acc parallel loop collapse(4)") \
    for (int i=0;i<grid.nx[0]; ++i) \
    for (int j=0;j<grid.nx[1]; ++j) \
    for (int k=0;k<grid.nx[2]; ++k) \
    for (int v=0;v<grid.nv; ++v)

#define FORALL(i,j,k,v) \
    for (int i=0;i<grid.nx[0]; ++i) \
    for (int j=0;j<grid.nx[1]; ++j) \
    for (int k=0;k<grid.nx[2]; ++k) \
    for (int v=0;v<grid.nv; ++v)


typedef struct Vars {
    real* ee;
    real* xx;
    real* ex_re;
    real* ex_im;
    real* bee;
    real* bxx;
    real* bex_re;
    real* bex_im;

    Vars(int size) {
        ee     = new real[size](); // all init to zero
        xx     = new real[size]();
        ex_re  = new real[size]();
        ex_im  = new real[size]();
        bee    = new real[size]();
        bxx    = new real[size]();
        bex_re = new real[size]();
        bex_im = new real[size]();
        #pragma acc enter data create(this,ee[0:size],xx[0:size],ex_re[0:size],ex_im[0:size],bee[0:size],bxx[0:size],bex_re[0:size],bex_im[0:size])
    }
    ~Vars() {
        #pragma acc exit data delete(ee, xx, ex_re, ex_im, bee, bxx, bex_re, bex_im, this)
        delete[] ee;
        delete[] xx;
        delete[] ex_re;
        delete[] ex_im;
        delete[] bee;
        delete[] bxx;
        delete[] bex_re;
        delete[] bex_im;
    }
} FieldVar;

typedef struct SnapShot_struct {
    std::list<real*> var_list;
    string fntpl;
    int every;
    std::vector<int> x_slices;   // coordinate for the reduced dimension
    std::vector<int> v_slices;

    // init with specified v-coordinate
    SnapShot_struct(std::list<real*> var_list_, string fntpl_, int every_,  std::vector<int> v_slices_) {
        var_list = var_list_;
        fntpl = fntpl_;
        every = every_;
        v_slices = v_slices_;
    }
    // init with specified y and v-coordinate
    SnapShot_struct(std::list<real*> var_list_, string fntpl_, int every_, std::vector<int> x_slices_, std::vector<int> v_slices_) {
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
        const int nvar = 8;
        int myrank = 0;
        real phy_time;
        real dt, dx;

        CartGrid grid;   // grid geometry of a MPI rank   // FIXME: why uniqie_ptr fail on MPI?
        real ds_L;       // = dx*dz/(z1-z0)/(x1-x0)

        FieldVar *v_stat, *v_rhs, *v_pre, *v_cor;  // field variables
        FieldVar *v_stat0;   // NOT used.

        real *P1,  *P2,  *P3,  *dN,  *dP;
        real *P1b, *P2b, *P3b, *dNb, *dPb;
        real *G0,*G0b;
        real n_nue0[2];   // initial number density for nue/nueb

        real CFL;
        real ko;

        const real theta = 37 * M_PI / 180.;  //1e-6;
        const real ct = cos(2*theta);
        const real st = sin(2*theta);
        real pmo = 0.1;      // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
        real mu  = 1.0;      // can be set by set_mu()
        bool renorm = false;  // can be set by set_renorm()

        std::ofstream anafile;
        std::list<SnapShot> snapshots;

        NuOsc(int px_[], int nv_, const int nphi_, const int gx_[],
              const real bbox[][2], const real dx_, const real CFL_, const real  ko_);

        ~NuOsc() {
            #pragma acc exit data delete(G0,G0b,P1,P2,P3,P1b,P2b,P3b,dP,dN,dPb,dNb)
            delete[] G0;
            delete[] G0b;
            delete[] P1;  delete[] P2;  delete[] P3;  delete[] dP;  delete[] dN;
            delete[] P1b; delete[] P2b; delete[] P3b; delete[] dPb; delete[] dNb;
            #pragma acc exit data delete(v_stat, v_rhs, v_pre, v_cor, v_stat0)
            delete v_stat, v_rhs, v_pre, v_cor, v_stat0;

            anafile.close();

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
        ulong get_lpts() const {  return grid.get_lpts();  }
        int  get_nv() const {  return grid.get_nv();  }

        void fillInitValue(int ipt, real alpha, real eps0, real sigma, real lnue, real lnueb, real lnuex, real lnuebx);
        void fillInitGaussian(real eps0, real sigma);
        void fillInitSquare(real eps0, real sigma);
        void fillInitTriangle(real eps0, real sigma);
        void updatePeriodicBoundary (FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);
        void step_rk4();
        void calRHS(FieldVar* out, const FieldVar * in);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2);
        void vectorize(FieldVar* v0, const FieldVar * v1, const real a, const FieldVar * v2, const FieldVar * v3);
        void analysis();
        void eval_conserved(const FieldVar* v0);
        void renormalize(const FieldVar* v0);

        void pack_buffer(const FieldVar* v0);
        void unpack_buffer(FieldVar* v0);
        void sync_boundary(FieldVar* v0);

        // 1D output:
        void addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int>  vidx);
        void checkSnapShot(const int t=0) const;
        // 2D output:
        void addSnapShotAtXV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> xidx, std::vector<int> vidx);

};
