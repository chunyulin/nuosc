#pragma once

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

#include "Point.h"

typedef double real;
typedef vector<real> Vec;
typedef vector<uint> IVec;

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
class IcosahedronVoronoi {
    public:
        const uint NP = 5;
        const IVec IB = { 1, 0, 2,  6,   6, 2, 7, 11,   2, 0, 3,  7,   7, 3, 8, 11,  3, 0, 4,  8, 
            8, 4, 9, 11,   4, 0,  5,  9,   9, 5, 10, 11,   5, 0, 1, 10,  10, 1, 6, 11  };
        const real lambda = 1.10714871779409;

        uint N;        // # unique points
        uint nn;       // # cell at the sides of each patch
        vector<Point> X;    // non-unique points in each patch
        Vec vw;

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
            bp[0]      = Point(0,0, 1);
            bp[2*NP+1] = Point(0,0,-1);

            // refined coordinates in based patch
            X.reserve(N);
            X[0]   = bp[0];
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

        IcosahedronVoronoi(uint nn_) : nn(nn_) {

            N = 2*NP*nn*nn+2;

            BasePoints();

            vw = Vec(N, 0);
            ui.reserve(2*NP*(nn+1)*(nn+1));  // index for unique points
            for (int p=0;p<2*NP;++p) _IcosahedronUniqueIndex(p);
            for (int p=0;p<2*NP;++p) _IcosahedronRefine(p);

        }

        //~IcosahedronVoronoi() {
        //}

    private:

        vector<Point> bp;   // 2*NP+2 base points (12 for Icosahedral, north/south pole for the first/last) 
        vector<uint>  ui;   // index for unique points

        inline uint fid(uint p, uint i, uint j) {  return (p*(nn+1) + i)*(nn+1) + j;     }

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
        }

        void _IcosahedronRefine(uint p) {

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
                    }
                } // end for ij
        }
};

