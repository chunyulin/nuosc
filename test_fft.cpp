#include <cmath>
#include <iostream>
#include <vector>
#include <complex>

#include <fftw3.h>

using std::exp;
using std::sin;
using std::cos;
using std::cout;
using std::endl;
using std::complex;
using std::vector;


int main(int argc, char *argv[]) {

    int N = 512;
    vector<double> x(N);
    vector<double> re(N);
    vector<double> im(N);
    
    vector<complex<double>> u1(N);
    vector<complex<double>> u2(N);


    double dx = 2.0/N;
    double x1 = 2*M_PI;
    double sigma = x1/5.0;
    for (int i=0;i<N;i++) {
        x[i] = -x1 * (0.5+i)*dx;
        //re[i] = sin(x[i]);
        //im[i] = cos(x[i]);
        re[i] = 0;
        im[i] = exp(-x[i]*x[i]/(2*sigma*sigma));
        u1[i] = {re[i], im[i]};
    }

    fftw_complex *in, *out, *tmp;
    fftw_plan p;

    in  = reinterpret_cast<fftw_complex*>(&u1[0]);
    out = reinterpret_cast<fftw_complex*>(&u2[0]);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


    fftw_execute(p); /* repeat as needed */




    for (int i=0;i<N;i++) {
        cout << in[i][0] << " " << in[i][1] << " " << out[i][0] << " " << out[i][1] << endl;
    }


    //fftw_destroy_plan(p);
    //fftw_free(in); fftw_free(out);



    return 0;
}
