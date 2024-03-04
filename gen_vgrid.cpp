#include "nuosc_class.h"
#include "icosahedron.h"

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
