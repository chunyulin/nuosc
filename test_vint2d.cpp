#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

using std::sqrt;
using std::abs;
using std::exp;
using std::sin;
using std::cos;
using std::cout;
using std::endl;
using std::vector;
using std::string;

#include "Point.h"

typedef double real;
typedef vector<real> Vec;
typedef vector<uint> IVec;

class Sphere {
    public:
        uint N;
        real exact;
        Vec vx,vy,vz,vw, f;

        Sphere() {};

        void init_f_s4() {
            #pragma omp parallel for
            for (uint i=0;i<N; ++i)  f[i] = vx[i]*vy[i]*vz[i]*vx[i]*vy[i]*vz[i];  // exact: 4 * M_PI /105
            exact = 4.0 * M_PI / 5005.0;
        }
        void init_f_s2() {
            #pragma omp parallel for
            for (uint i=0;i<N; ++i)  f[i] = vx[i]*vy[i]*vz[i]*vx[i]*vy[i]*vz[i];  // exact: 4 * M_PI /105
            exact = 4.0 * M_PI / 105.0;
        }
        void init_f_s0() {
            #pragma omp parallel for
            for (uint i=0;i<N; ++i)  f[i] = 1.0;
            exact = 4.0 * M_PI;
        }
        void init_f() { init_f_s2();  }

        real integral() {
            real sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (uint i=0;i<N; ++i) {
                sum += f[i]*vw[i];
            }
            return sum;
        }
};

/****
// generate refined NxN elements for each patch, with cornor(A,B,C,D).
//     B   N
// j n/ \ / \
//   A 0 D 2 x 4...
// i n\ / \ / \
//     C 1 x 3 x 5 ...
//      \ / \ / 
//       S   x
**/
class IcosahedronVoronoi : public Sphere {
    public:
        const uint NP = 5;
        const IVec IB = { 1, 0, 2,  6,   6, 2, 7, 11,   2, 0, 3,  7,   7, 3, 8, 11,  3, 0, 4,  8, 
                          8, 4, 9, 11,   4, 0,  5,  9,   9, 5, 10, 11,   5, 0, 1, 10,  10, 1, 6, 11  };
        const real lambda = 1.10714871779409;
        uint nn;

        void BasePoints() {
            bp.reserve(2*NP+2);
            for (int i=1; i<=NP; ++i) {
                bp[i].x = cos(2*i*M_PI/NP)*sin(lambda);
                bp[i].y = sin(2*i*M_PI/NP)*sin(lambda);
                bp[i].z = cos(lambda);
                bp[i+NP].x = cos((2*i+1)*M_PI/NP)*sin(lambda);
                bp[i+NP].y = sin((2*i+1)*M_PI/NP)*sin(lambda);
                bp[i+NP].z = -bp[i].z;
            }
            bp[0]      = Point(0,0,1);
            bp[2*NP+1] = Point(0,0,-1);
            
            // refined coordinates in based patch
            X.reserve(N);   // all unique Delaunay points
            X[0]    = bp[0];
            X[N-1] = bp[2*NP+1];
            #pragma omp parallel for
            for (int p=0; p<2*NP; ++p) {
                // cornor of the basis patch.
                uint a = IB[  4*p];
                uint b = IB[1+4*p];
                uint c = IB[2+4*p];
                uint d = IB[3+4*p];

                for (int i=0; i<nn; ++i)
                for (int j=0; j<nn; ++j) {
                    uint pij = p*nn*nn + i*nn + j + 1;
                    X[pij] = i*(j*bp[c]+(nn-j)*bp[d]) + (nn-i)*(j*bp[b]+(nn-j)*bp[a]);
                    real iL = 1.0 / X[pij].norm();
                    X[pij] = iL * X[pij];
                }
            }
        }

        void dumpBasePoints() {
    	    for(int i=0;i<8*NP;++i)  bb << bp[IB[i]] << endl;
        }
        
        void showIcosaExact() {
            real a2 = 2*(1.0 - cos(lambda));
            printf("Exact number for icosahedron: a = %.10g  Surface area = %.10g\n", sqrt(a2), 5.0*std::sqrt(3.0)*a2);
        }

        IcosahedronVoronoi(uint nn_) : nn(nn_) {

            N = 2*NP*nn*nn+2;

            pp.open("p.dat", std::ofstream::out | std::ofstream::trunc);
            bb.open("b.dat", std::ofstream::out | std::ofstream::trunc);
            vv.open("v.dat", std::ofstream::out | std::ofstream::trunc);

            BasePoints();

            vx.reserve(N);
            vy.reserve(N);
            vz.reserve(N);
            f.reserve(N);
            vw = Vec(N, 0);
            ui.reserve(2*NP*(nn+1)*(nn+1));  // index for unique points

            for (int p=0;p<2*NP;++p) _IcosahedronUniqueIndex(p);
            for (int p=0;p<2*NP;++p) _IcosahedronRefine(p);

            init_f();
        }

        ~IcosahedronVoronoi() {
            pp.close();
            bb.close();
            vv.close();
        }
        
    private:

        vector<Point> bp;   // 2*NP+2 base points (12 for Icosahedral, north/south pole for the first/last) 
        vector<Point> X;    // unique points in each patch
        vector<uint>  ui;   // index for unique points

        std::ofstream pp, bb, vv;

        inline uint fid(uint p, uint i, uint j) {  return (p*(nn+1) + i)*(nn+1) + j;     }
        inline uint sid(uint p, uint i, uint j) {  return (p*nn + i)*nn + j;     }

        void _IcosahedronUniqueIndex(uint p) {

            // calculate index for accesssing unique points
            #pragma omp parallel for collapse(2)
            for (int i=0; i<nn; ++i)
            for (int j=0; j<nn; ++j) {
                ui[fid(p,i,j)] = p*nn*nn + i*nn + j + 1;
            }
            
            if (p==2*NP-1) {     // last patch
                for (int i=1; i<=nn; ++i) {
                   ui[fid(p, i-1, nn)] =               1 + nn*(i-1);            // upper-right side
                   ui[fid(p,  nn,  i)] = 2*nn*nn - (nn-1) - nn*(i-1);      // bottom-right side
                }
                ui[fid(p,nn,0)] = N-1;  // S-pole
            } else if (1==p%2)  {   // lower NP patched
                for (int i=1; i<=nn; ++i) {
                   ui[fid(p, i-1, nn)] = (p+1)*nn*nn +      1 + nn*(i-1);  // upper-right side
                   ui[fid(p,  nn,  i)] = (p+3)*nn*nn - (nn-1) - nn*(i-1);  // bottom-right side
                }
                ui[fid(p,nn,0)] = N-1;  // S-pole
            } else {                // upper NP patched
                for (int i=1; i<=nn; ++i) {
                   ui[fid(p, i,nn)]     = ((p+2)*nn*nn + (nn- i) + 1) % (2*NP*nn*nn);  // upper-right side
                   ui[fid(p,  nn, i-1)] =  (p+1)*nn*nn + i;                     // bottom-right side
                }
                ui[fid(p,0,nn)] = 0;     // N-pole
            }
            
            #if 0
            for (int i=0; i<nn+1; ++i)
            for (int j=0; j<nn+1; ++j) {
                uint pij = fid(p,i,j);
                printf("p=%2d i=%3d j=%3d:  pij=%5d ui[pij]=%5d   (%6f %6f %6f) \n", p, i, j, pij, ui[pij], X[ui[pij]].x, X[ui[pij]].y, X[ui[pij]].z);
            }
            printf("\n");
            #endif
        }

        void _IcosahedronRefine(uint p) {

            // assign v
            #pragma omp parallel for
            for (int i=0; i<N; ++i) {
                vx[i] = X[i].x;
                vy[i] = X[i].y;
                vz[i] = X[i].z;
            }

            //#pragma omp parallel for collapse(2)
            for (int i=0; i<nn; ++i)
            for (int j=0; j<nn; ++j) {

                auto iA = ui[fid(p,i  ,j  )];
                auto iC = ui[fid(p,i+1,j+1)];
                auto A = X[iA];
                auto C = X[iC];
                {
                auto iB = ui[fid(p,i  ,j+1)];
                auto B = X[iB];
                auto O = circumcenter( A,B,C );
                auto fAB = ((O-A)*(B-A) * 0.25).norm();
                auto fBC = ((O-B)*(C-B) * 0.25).norm();
                auto fCA = ((O-C)*(A-C) * 0.25).norm();
                vw[ iA ] += fAB + fCA;
                vw[ iB ] += fBC + fAB;
                vw[ iC ] += fCA + fBC;
            #if 0
                pp << A << endl << B << endl << C << endl << O << endl;
            #endif
                }
                {
                auto iB = ui[fid(p,i+1,j)];
                auto B = X[iB];
                auto O = circumcenter( A,B,C );
                auto fAB = ((O-A)*(B-A) * 0.25).norm();
                auto fBC = ((O-B)*(C-B) * 0.25).norm();
                auto fCA = ((O-C)*(A-C) * 0.25).norm();
                vw[ iA ] += fAB + fCA;
                vw[ iB ] += fBC + fAB;
                vw[ iC ] += fCA + fBC;
            #if 0
                pp << A << endl << B << endl << C << endl << O << endl;
            #endif
                }
            #if 0
                vv << vx[iA] << " "  << vy[iA] << " "  << vz[iA] << " "  << vw[iA] << endl;
            #endif
            } // end for ij
        }
};


class SpherePolarCC : public Sphere {
    public:
        uint nt, np;

        SpherePolarCC(uint nt_, uint np_) : nt(nt_), np(np_) {
            N = nt*np;
            vx.reserve(N);
            vy.reserve(N);
            vz.reserve(N);
            vw.reserve(N);
            f.reserve(N);

            _polargrid();
            init_f();
        }

    private:
        void _polargrid() {
            real dt = M_PI/nt;
            real dp = 2.0*M_PI/np;
            #pragma omp parallel for collapse(2)
            for (uint i=0;i <nt; i++)
            for (uint j=0;j <np; j++) {
                    auto idx = i*np+j;
                    real tet = (i+0.5)* dt;
                    real phi = (j+0.5)* dp;
                    vx[idx] = sin(tet)*cos(phi);
                    vy[idx] = sin(tet)*cos(phi);
                    vz[idx] = cos(tet);
                    vw[idx] = sin(tet)*dt*dp;
            }
        }
};

/****
// generate refined MxN pts at each patch, with cornor(A,B,C,D).
//    B   B
//  m/ \ / \
//  A   C   x...
//  n\ / \ / \
//    D   x   x
//     \ / \ / 
//      S   S
//
// NP=3  1.23095941734077
// NP=4  1.14371774040242
// NP=5  1.10714871779409
// NP=6  1.08817621336417
// NP=7  1.07702232077537
****/

class IcosahedronDelaunay  : public Sphere {
    public:
        const uint NP = 5;
        /*const IVec IB0 = { 1, 0, 2, 6,   6, 2,  7, 11,
                          2, 0, 3, 7,   7, 3,  8, 11,
                          3, 0, 4, 8,   8, 4,  9, 11,
                          4, 0, 5, 9,   9, 5, 10, 11,
                          5, 0, 1,10,  10, 1,  6, 11  }; */
        const IVec IB = { 1, 0, 2,  6,    2, 0, 3,  7,    3, 0, 4,  8,    4, 0,  5,  9,    5, 0, 1, 10,
                          6, 2, 7, 11,    7, 3, 8, 11,    8, 4, 9, 11,    9, 5, 10, 11,   10, 1, 6, 11  };
        const real lambda = 1.10714871779409;
        uint nm, nn;
        Vec x,y,z;

        Vec x0, y0, z0;  // 2*NP+2 base points (12 for Icosahedral, north/south pole for the first/last) 
        void BasePoints() {
            x0.reserve(2*NP+2);
            y0.reserve(2*NP+2);
            z0.reserve(2*NP+2);
            for (int i=1; i<=NP; ++i) {
                x0[i] = cos(2*i*M_PI/NP)*sin(lambda);
                y0[i] = sin(2*i*M_PI/NP)*sin(lambda);
                z0[i] = cos(lambda);
                x0[i+NP] = cos((2*i+1)*M_PI/NP)*sin(lambda);
                y0[i+NP] = sin((2*i+1)*M_PI/NP)*sin(lambda);
                z0[i+NP] = -z0[i];
            }
            x0[0] = 0;    x0[2*NP+1] = 0;
            y0[0] = 0;    y0[2*NP+1] = 0;
            z0[0] = 1.0;  z0[2*NP+1] = -1.0;
        }

        void dumpBasePoints() {
    	    for(int i=0;i<8*NP;++i) {
    	         bb << x0[IB[i]] << " " << y0[IB[i]] << " " << z0[IB[i]] << endl;
    	    }
        }
        
        void showIcosaExact() {
            real a2 = 2*(1.0 - cos(lambda));
            printf("Exact number for icosahedron: a = %.10g  Surface area = %.10g\n", sqrt(a2), 5.0*std::sqrt(3.0)*a2);
        }

        IcosahedronDelaunay (uint nm_, uint nn_) : nn(nn_), nm(nm_) {

            pp.open("p.dat", std::ofstream::out | std::ofstream::trunc);
            bb.open("b.dat", std::ofstream::out | std::ofstream::trunc);
            vv.open("v.dat", std::ofstream::out | std::ofstream::trunc);

            BasePoints();

            N = 4*nm*nn*NP;   // # of triangle elements
            vx.reserve(N);
            vy.reserve(N);
            vz.reserve(N);
            vw.reserve(N);
            f.reserve(N);
            
            for (int p=0;p<2*NP;++p) {
                _IcosahedronRefine(p);
            }
            init_f();
        }

        ~IcosahedronDelaunay () {
        pp.close();
        bb.close();
        vv.close();
        }
        
    private:
        std::ofstream pp, bb, vv;
        void _IcosahedronRefine(uint p) {

            uint a = IB[  4*p];
            uint b = IB[1+4*p];
            uint c = IB[2+4*p];
            uint d = IB[3+4*p];

            Vec x((nm+1)*(nn+1));
            Vec y((nm+1)*(nn+1));
            Vec z((nm+1)*(nn+1));

            #pragma omp parallel for collapse(2)
            for (int j=0; j<nn+1; ++j)
            for (int i=0; i<nm+1; ++i)  {
                uint ij = j*(nm+1) + i;
                x[ij] = i*(j*x0[c]+(nn-j)*x0[b]) + (nm-i)*(j*x0[d]+(nn-j)*x0[a]);
                y[ij] = i*(j*y0[c]+(nn-j)*y0[b]) + (nm-i)*(j*y0[d]+(nn-j)*y0[a]);
                z[ij] = i*(j*z0[c]+(nn-j)*z0[b]) + (nm-i)*(j*z0[d]+(nn-j)*z0[a]);
                real iL = 1.0 / std::sqrt(x[ij]*x[ij] + y[ij]*y[ij] + z[ij]*z[ij]);
                x[ij] *= iL;
                y[ij] *= iL;
                z[ij] *= iL;
            }

            #pragma omp parallel for collapse(2)
            for (int j=0; j<nn; ++j)
            for (int i=0; i<nm; ++i)  {
                uint ij  = j*(nm+1) + i;            // index of x,y,z
                
                {   uint pij = 2*(p*nn*nm + j*nm + i);  // index of vx,vy,vz
                real ax = x[ij+nm+2] - x[ij];
                real ay = y[ij+nm+2] - y[ij];
                real az = z[ij+nm+2] - z[ij];
                real bx = x[ij+1] - x[ij];
                real by = y[ij+1] - y[ij];
                real bz = z[ij+1] - z[ij];
                real axb = 0.5* std::sqrt( (ay*bz-az*by)*(ay*bz-az*by) + (az*bx-ax*bz)*(az*bx-ax*bz) + (ax*by-ay*bx)*(ax*by-ay*bx) );
                vw[pij] = axb;
                
                vx[pij] = x[ij] + x[ij+nm+2] + x[ij+1];
                vy[pij] = y[ij] + y[ij+nm+2] + y[ij+1];
                vz[pij] = z[ij] + z[ij+nm+2] + z[ij+1];
                real tmp = 1.0 / std::sqrt(vx[pij]*vx[pij] +vy[pij]*vy[pij]+vz[pij]*vz[pij]);
                vx[pij] *= tmp;
                vy[pij] *= tmp;
                vz[pij] *= tmp;
                
                #if 0
                vv << vx[pij] << " " << vy[pij] << " " << vz[pij] << " " << vw[pij] << endl;
                real La = std::sqrt(ax*ax+ay*ay+az*az);
                real Lb = std::sqrt(bx*bx+by*by+bz*bz);
                real tt = (bx*bx+by*by+bz*bz);
                real cosab = (ax*bx+ay*by+az*bz)/La/Lb;
                real sinab = 2*axb/La/Lb;
                printf("|a|=%.10g |b|=%.10g cosAB=%g |AxB|/2=%.10g vw=%.10g 1=%g\n", La, Lb, 
                    cosab, La*Lb*sqrt(3.0)*0.25, vw[pij], cosab*cosab+sinab*sinab);
                #endif
                }
                
                {   uint pij = 2*(p*nn*nm + j*nm + i) + 1;  // index of vx,vy,vz
                real ax = x[ij+nm+2] - x[ij];
                real ay = y[ij+nm+2] - y[ij];
                real az = z[ij+nm+2] - z[ij];

                real bx = x[ij+nm+1] - x[ij];
                real by = y[ij+nm+1] - y[ij];
                real bz = z[ij+nm+1] - z[ij];
                real axb = 0.5* std::sqrt( (ay*bz-az*by)*(ay*bz-az*by) + (az*bx-ax*bz)*(az*bx-ax*bz) + (ax*by-ay*bx)*(ax*by-ay*bx) );
                vw[pij] = axb;

                vx[pij] = x[ij] + x[ij+nm+2] + x[ij+nm+1];
                vy[pij] = y[ij] + y[ij+nm+2] + y[ij+nm+1];
                vz[pij] = z[ij] + z[ij+nm+2] + z[ij+nm+1];
                real tmp = 1.0 / std::sqrt(vx[pij]*vx[pij] +vy[pij]*vy[pij]+vz[pij]*vz[pij]);
                vx[pij] *= tmp;
                vy[pij] *= tmp;
                vz[pij] *= tmp;

                #if 0
                vv << vx[pij] << " " << vy[pij] << " " << vz[pij] << " " << vw[pij] << endl;

                real La = std::sqrt(ax*ax+ay*ay+az*az);
                real Lb = std::sqrt(bx*bx+by*by+bz*bz);
                real cosab = (ax*bx+ay*by+az*bz)/La/Lb;
                real sinab = 2*axb/La/Lb;
                printf("|a|=%.10g |b|=%.10g cosAB=%g |AxB|/2=%.10g vw=%.10g 1=%g\n", La, Lb, 
                    cosab, La*Lb*sqrt(3.0)*0.25, vw[pij], cosab*cosab+sinab*sinab);
                #endif

                }
            }
        }
};

class IcosahedronSquare : public Sphere {
    public:
        const uint NP = 5;
        const IVec IB = { 1, 0, 2,  6,   2, 0,  3, 7,   3, 0, 4, 8,    4, 0, 5, 9,     5, 0, 1,10,
                          6, 2, 7, 11,   7, 3,  8, 11,  8, 4,  9, 11,  9, 5, 10, 11,   10, 1,  6, 11  };
        const real lambda = 1.10714871779409;
        uint nm, nn;
        Vec x,y,z;

        Vec x0, y0, z0;  // 2*NP+2 base points (12 for Icosahedral, north/south pole for the first/last) 
        void BasePoints() {
            x0.reserve(2*NP+2);
            y0.reserve(2*NP+2);
            z0.reserve(2*NP+2);
            for (int i=1; i<=NP; ++i) {
                x0[i] = cos(2*i*M_PI/NP)*sin(lambda);
                y0[i] = sin(2*i*M_PI/NP)*sin(lambda);
                z0[i] = cos(lambda);
                x0[i+NP] = cos((2*i+1)*M_PI/NP)*sin(lambda);
                y0[i+NP] = sin((2*i+1)*M_PI/NP)*sin(lambda);
                z0[i+NP] = -z0[i];
            }
            x0[0] = 0;    x0[2*NP+1] = 0;
            y0[0] = 0;    y0[2*NP+1] = 0;
            z0[0] = 1.0;  z0[2*NP+1] = -1.0;
        }

        void showIcosaExact() {
            real a2 = 2*(1.0 - cos(lambda));
            printf("Exact number for icosahedron: a = %.10g  Surface area = %.10g\n", sqrt(a2), 5.0*std::sqrt(3.0)*a2);
        }

        IcosahedronSquare(uint nm_, uint nn_) : nn(nn_), nm(nm_) {

            BasePoints();

            N = 2*nm*nn*NP;   // # of triangle elements
            vx.reserve(N);
            vy.reserve(N);
            vz.reserve(N);
            vw.reserve(N);
            f.reserve(N);
            
            for (int p=0;p<2*NP;++p) {
                _IcosahedronRefine(p);
            }
            init_f();
        }

    private:
        void _IcosahedronRefine(uint p) {

            uint a = IB[  4*p];
            uint b = IB[1+4*p];
            uint c = IB[2+4*p];
            uint d = IB[3+4*p];

            Vec x((nm+1)*(nn+1));
            Vec y((nm+1)*(nn+1));
            Vec z((nm+1)*(nn+1));

            #pragma omp parallel for collapse(2)
            for (int j=0; j<nn+1; ++j)
            for (int i=0; i<nm+1; ++i)  {
                uint ij = j*(nm+1) + i;
                x[ij] = i*(j*x0[c]+(nn-j)*x0[b]) + (nm-i)*(j*x0[d]+(nn-j)*x0[a]);
                y[ij] = i*(j*y0[c]+(nn-j)*y0[b]) + (nm-i)*(j*y0[d]+(nn-j)*y0[a]);
                z[ij] = i*(j*z0[c]+(nn-j)*z0[b]) + (nm-i)*(j*z0[d]+(nn-j)*z0[a]);
                real iL = 1.0 / std::sqrt(x[ij]*x[ij] + y[ij]*y[ij] + z[ij]*z[ij]);
                x[ij] *= iL;
                y[ij] *= iL;
                z[ij] *= iL;
            }

            #pragma omp parallel for collapse(2)
            for (int j=0; j<nn; ++j)
            for (int i=0; i<nm; ++i)  {
                uint ij  = j*(nm+1) + i;            // index of x,y,z
                
                {   uint pij = (p*nn*nm + j*nm + i);  // index of vx,vy,vz
                real ax = x[ij+1] - x[ij];
                real ay = y[ij+1] - y[ij];
                real az = z[ij+1] - z[ij];
                real bx = x[ij+nm+1] - x[ij];
                real by = y[ij+nm+1] - y[ij];
                real bz = z[ij+nm+1] - z[ij];
                real axb = std::sqrt( (ay*bz-az*by)*(ay*bz-az*by) + (az*bx-ax*bz)*(az*bx-ax*bz) + (ax*by-ay*bx)*(ax*by-ay*bx) );
                vw[pij] = axb;
                
                vx[pij] = x[ij] + x[ij+1] + x[ij+nm+1] + x[ij+nm+2];
                vy[pij] = y[ij] + y[ij+1] + y[ij+nm+1] + y[ij+nm+2];
                vz[pij] = z[ij] + z[ij+1] + z[ij+nm+1] + z[ij+nm+2];
                real tmp = 1.0 / std::sqrt(vx[pij]*vx[pij] +vy[pij]*vy[pij]+vz[pij]*vz[pij]);
                vx[pij] *= tmp;
                vy[pij] *= tmp;
                vz[pij] *= tmp;
                }
            }
        }
};


int main(int argc, char *argv[]) {

    const int N=2000;

    std::chrono::time_point<std::chrono::high_resolution_clock> t1;

#if 1
    for (uint n=2; n<N; n*=2) {
       t1 = std::chrono::high_resolution_clock::now();
       SpherePolarCC sgpcc(n,2*n);
       real Ipcc = sgpcc.integral();
       real stepms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count();
       printf("dIpcc for n=%5d : %.10g -- Time: %g ms\n", n, Ipcc-sgpcc.exact, stepms);
    }
#endif

#if 1
    for (uint n=1; n<N; n<<=1) {
       t1 = std::chrono::high_resolution_clock::now();
       IcosahedronDelaunay  icosa(n,n);
       if (n==1) {
          icosa.showIcosaExact();
          icosa.dumpBasePoints();
       }
       real integral = icosa.integral();
       real stepms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count();

        real max = -10, min=100;
        for(int i=0;i<icosa.N;++i) {
            max = icosa.vw[i]>max? icosa.vw[i] : max;
            min = icosa.vw[i]<min? icosa.vw[i] : min;
        }

       printf("IcosahedronDelaunay, n = %5d, I= %10.5g (%10.7g),  dVe= %4.1f%  Time: %g ms\n", n, integral, icosa.exact- integral,  (max-min)/max*100, stepms);
    }
#endif

#if 1
    for (uint n=1; n<N; n<<=1) {
       t1 = std::chrono::high_resolution_clock::now();
       IcosahedronVoronoi icosa(n);
       if (n==1) {
          icosa.showIcosaExact();
          icosa.dumpBasePoints();
       }
       real integral = icosa.integral();
       real stepms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count();

        real max = -10, min=100;
        for(int i=0;i<icosa.N;++i) {
            max = icosa.vw[i]>max? icosa.vw[i] : max;
            min = icosa.vw[i]<min? icosa.vw[i] : min;
        }

       printf("IcosahedronVoronoi, n = %5d, I= %10.5g (%10.7g),  dVe= %4.1f%  Time: %g ms\n", n, integral, icosa.exact- integral,  (max-min)/max*100, stepms);
    }
#endif

#if 1
    for (uint n=1; n<N; n<<=1) {
       t1 = std::chrono::high_resolution_clock::now();
       IcosahedronSquare icosa(n,n);
       real Iicosa = icosa.integral();
       real stepms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count();
       printf("IcosahedronSquare, n = %5d, I= %15.10g (%15.10g)    Time: %g ms\n", n, Iicosa, icosa.exact-Iicosa, stepms);
    }
#endif
    return 0;
}
