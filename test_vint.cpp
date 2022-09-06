#include <cmath>
#include <iostream>
#include <vector>
#include "jacobi_poly.h"

using std::exp;
using std::sin;
using std::cos;
using std::cout;
using std::endl;
using std::vector;

typedef vector<double> Vec;

class Disk {
    public:
        int nvr, nvt;       // # of v cubature points.
        int nv;       // # of v cubature points.
        Vec vw;                 // integral quadrature
        Vec vz;
        Vec vy;
        Vec f;
        double sum;

        Disk(int nvr_, int nvt_) { 
            nvr = nvr_;
            nvt = nvt_;
            vy.reserve(nvr*nvt);
            vz.reserve(nvr*nvt);
            vw.reserve(nvr*nvt);
            f.reserve(nvr*nvt);
            //polar_grid(nvr, nvt);
        }

        Disk(int nxv_) { 
            nv = nxv_;
            vy.reserve(nv*nv);
            vz.reserve(nv*nv);
            vw.reserve(nv*nv);
            f.reserve(nv*nv);
            //elliptical_grid(nv);
        }

        void setp() {
            for (int i=0;i <nvt; ++i)
            for (int j=0;j <nvr; ++j) {
                    auto idx = i*nvr+j;
                    f[idx] = (vy[idx]*vy[idx]);
            }
        }

        void sete() {
            for (int i=0;i <nv; ++i)
                for (int j=0;j <nv; ++j) {
                    auto idx = i*nv+j;
                    f[idx] = (vy[idx]*vy[idx]);
                }
        }

        void integrale() {
            sum = 0;
            for (int i=0;i <nv; ++i)
                for (int j=0;j <nv; ++j) {
                    auto idx = i*nv+j;
                    sum += f[idx]*vw[idx];
                }
        }

        void integralp() {
            sum = 0;
            for (int i=0;i <nvt; ++i)
                for (int j=0;j <nvr; ++j) {
                    auto idx = i*nvr+j;
                    sum += f[idx]*vw[idx];
                }
        }


        void polar_grid() {
            double dvr = 1.0/(nvr);
            double dvt = 2.0*M_PI/(nvt);
            for (int i=0;i <nvt; i++)
                for (int j=0;j <nvr; j++) {
                    auto idx = i*nvr+j;
                    double r = (j+0.5)*dvr;
                    double t = (i+0.5) *dvt;
                    vy[idx] = r *cos(t);
                    vz[idx] = r *sin(t);
                    vw[idx] = dvr*dvt*r;
                }
        }

        void elliptical_grid() {
            double dv = 2.0/(nv-1);
            for (int i=0;i<nv; i++)
                for (int j=0;j<nv; j++) {
                    auto idx = i*nv+j;
                    double y = i*dv-1;
                    double z = j*dv-1;
                    vy[idx] = y*sqrt(1.0-0.5*z*z);
                    vz[idx] = z*sqrt(1.0-0.5*y*y);
                    vw[idx] = (2-z*z-y*y)/sqrt((2-z*z)*(2-y*y))     *dv*dv/(1<<(int(i==0)+int(j==0)));
                }
        }

        void elliptical_GL_grid() {
            Vec r(nv);
            Vec w(nv);
            JacobiGL(nv-1,0,0,r,w);
            for (int i=0;i<nv; i++)
            for (int j=0;j<nv; j++) {
                auto idx = i*nv+j;
                double y = r[i];
                double z = r[j];
                vy[idx] = y*sqrt(1.0-0.5*z*z);
                vz[idx] = z*sqrt(1.0-0.5*y*y);
                vw[idx] = (2-z*z-y*y)/sqrt((2-z*z)*(2-y*y))   *w[i]*w[j];
            }
        }



        void FG_grid() {
            double dv = 2.0/(nv-1);
            for (int i=0;i<nv; i++)
                for (int j=0;j<nv; j++) {
                    double y = i*dv-1;
                    double z = j*dv-1;
                    double r = sqrt(y*y+z*z);
                    double m = sqrt(y*y-y*y*z*z+z*z);
                    auto idx = i*nv+j;
                    vy[idx] = y*m/(r+1.e-15);
                    vz[idx] = z*m/(r+1.e-15);
                    vw[idx] = (1.0-(2*z*z*y*y)/(z*z+y*y+1e-100)) * dv*dv/(1<<(int(i==0)+int(j==0)));
                }
        }

};



int main(int argc, char *argv[]) {

    const int N=1024;

#if 0    
    for (int i=2;i<=N;i*=2) {
        Disk fe(i+1);
        fe.FG_grid();
        fe.sete();
        fe.integrale();
        cout << (i+1)*(i+1) << " " <<fe.sum - 0.25*M_PI<< endl;
    }
#endif

    double old=0;
    for (int i=2;i<=N;i*=2) {
        Disk fe(i+1);
        fe.elliptical_grid();
        fe.sete();
        fe.integrale();
        cout << (i+1)*(i+1) << " " <<fe.sum - old<< endl;
        old = fe.sum;
    }


    for (int i=2;i<=N;i*=2) {
        Disk fe(i+1);
        fe.elliptical_GL_grid();
        fe.sete();
        fe.integrale();
        cout << (i+1)*(i+1) << " " <<fe.sum - old<< endl;
        old = fe.sum;
    }


    for (int i=2;i<=N;i*=2) {
        Disk fp(i,2*i);
        fp.polar_grid();
        fp.setp();
        fp.integralp();
        cout << i*i*2 << " " <<fp.sum  - old<< endl;
        old = fp.sum;
    }



    return 0;
}
