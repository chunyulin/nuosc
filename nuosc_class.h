#pragma once
#include "common.h"
#include "CartGrid.h"

#define COLLAPSE_LOOP 3

#define FORALL(i,j,v) \
    for (int i=0;i<grid.nx; ++i) \
    for (int j=0;j<grid.nz; ++j) \
    for (int v=0;v<grid.nv; ++v)

#define PARFORALL(i,j,v) \
    _Pragma("omp parallel for collapse(3)") \
    _Pragma("acc parallel loop collapse(3)") \
    FORALL(i,j,v)

typedef struct Vars {
    real *ee,  *mm, *tt;
    real *emr, *mtr, *ter;
    real *emi, *mti, *tei;
    real *bee, *bmm, *btt;
    real *bemr, *bmtr, *bter;
    real *bemi, *bmti, *btei;

    Vars(int size) {
        ee   = new real[size](); // all init to zero
        mm   = new real[size]();
        emr  = new real[size]();
        emi  = new real[size]();
        bee   = new real[size]();
        bmm   = new real[size]();
        bemr  = new real[size]();
        bemi  = new real[size]();
        #ifdef N_FLAVOR_3
        tt   = new real[size]();
        mtr  = new real[size]();
        mti  = new real[size]();
        ter  = new real[size]();
        tei  = new real[size]();
        btt   = new real[size]();
        bmtr  = new real[size]();
        bmti  = new real[size]();
        bter  = new real[size]();
        btei  = new real[size]();
        #endif
        //#pragma acc enter data create(this,ee[0:size],xx[0:size],ex_re[0:size],ex_im[0:size],bee[0:size],bxx[0:size],bex_re[0:size],bex_im[0:size])
    }
    ~Vars() {
        //#pragma acc exit data delete(ee, xx, ex_re, ex_im, bee, bxx, bex_re, bex_im, this)
        delete[] ee;   delete[] mm;   delete[] emr;   delete[] emi;
        delete[] bee;   delete[] bmm;   delete[] bemr;   delete[] bemi;
        #ifdef N_FLAVOR_3
        delete[] tt;   delete[] mtr;  delete[] ter;   delete[] mti;  delete[] tei;
        delete[] btt;   delete[] bmtr;  delete[] bter;   delete[] bmti;  delete[] btei;
        #endif
    }
    
    inline std::vector<real*> getAllFields() {
    #ifdef N_FLAVOR_3
    return std::vector<real*>{ee, mm, tt, emr, emi, mtr, mti, ter, tei,
                              bee, bmm, btt, bemr, bemi, bmtr, bmti, bter, btei};
    #else
    return std::vector<real*>{ee, mm, emr, emi, bee, bmm, bemr, bemi };
    #endif
    }

} FieldVar;

typedef struct SnapShot_struct {
    std	::list<real*> var_list;
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
        real phy_time, dt, dx, dz;

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

        real hee,hmm,htt,hemr,hemi,hmtr,hmti,hter,htei;
        const real theta = 37 * M_PI / 180.;  //1e-6;
        const real ct = cos(2*theta);
        const real st = sin(2*theta);

        real pmo = 0.1;      // 1 (-1) for normal (inverted) mass ordering, 0.0 for no vacuum term
        real mu  = 1.0;      // can be set by set_mu()
        bool renorm = false;  // can be set by set_renorm()

        std::ofstream anafile;
        std::list<SnapShot> snapshots;

        NuOsc(const int px_, const int pz_, const int nv_, const int nphi_, const int gx_,const int gz_,
                const real  x0_, const real  x1_, const real  z0_, const real  z1_, const real dx_, const real dz_, 
                const real CFL_, const real  ko_);


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
        void fillInitAdvtest(real eps0, real sigma);
        void updatePeriodicBoundary (FieldVar * in);
        void updateInjetOpenBoundary(FieldVar * in);

        void step_rk4();
        void step_srk3();
        void calRHS(FieldVar* out, const FieldVar * in);
        void vectorize(FieldVar* v0, FieldVar * v1, const real a, FieldVar *  v2);
        void vectorize(FieldVar* v0, FieldVar * v1, const real a, FieldVar *  v2, FieldVar * v3);
        void vectorize(FieldVar* v0, const real a, FieldVar * const v1, const real b, FieldVar * v2, const real dt, FieldVar * v3);

        void analysis();
        void eval_conserved(const FieldVar* v0);
        void renormalize(const FieldVar* v0);

        void pack_buffer(FieldVar* v0);
        void unpack_buffer(FieldVar* v0);
        void sync_boundary(FieldVar* v0);

        // 1D output:
        void addSnapShotAtV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int>  vidx);
        void checkSnapShot(const int t=0) const;
        // 2D output:
        void addSnapShotAtXV(std::list<real*> var, char *fntpl, int dumpstep, std::vector<int> xidx, std::vector<int> vidx);
};
