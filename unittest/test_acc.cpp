#include<cstdio>
#include<omp.h>
#include<iostream>
#include<openacc.h>
using std::cout;
using std::endl;


int main() {

            int NGPU = 1;
            omp_set_num_threads(2);

#if defined(_OPENACC)
            acc_device_t dev_type = acc_get_device_type();
            NGPU = acc_get_num_devices( dev_type );
            printf("OpenACC Enabled with %d GPU.\n", NGPU );
#endif

long nx=80000, ny=80000;
long dd = nx/NGPU;

cout << (nx*ny + 2*ny)*8/1e12 << " GB" << endl;

double *A = new double[nx*ny];
double *x = new double[ny];
double *b = new double[nx];

for (long i=0; i<nx*ny; i++) A[i] = 1.;
for (long i=0; i<ny; i++)    x[i] = 1.;
for (long i=0; i<nx; i++)    b[i] = 0.;
cout << "CPU x = " << x << endl;

#pragma omp parallel num_threads(NGPU)
{
   printf("OpenMP Enabled with %d / %d threads.\n",  omp_get_thread_num() ,  omp_get_num_threads() );
   int tid = omp_get_thread_num();
#if defined(_OPENACC)
   acc_set_device_num( omp_get_thread_num()+1, dev_type );
#endif
   #pragma acc update device (A[tid*dd*ny:(tid+1)*dd*ny],x[:ny],b[tid*dd:(tid+1)*dd])
   cout << tid << " " << x << endl;

   //cout << tid*dd*ny << " " << (tid+1)*dd*ny << endl;

   #pragma acc parallel loop
   for (long i=tid*dd; i<(tid+1)*dd; i++) {
     double t = 0;
     #pragma acc loop reduction (+:t)
     for (int j=0; j<ny; j++) {
       t += A[i*ny+j]*x[j];
     }
     b[i] = t;

   }
}


std::cout << b[10] << " " << b[150] << " " << b[250] << std::endl;   // 0 0 0

#pragma acc update host (b[:100], b[200:300])
std::cout << b[10] << " " << b[150] << " " << b[250] << std::endl;   // 1000 0 1000


#pragma acc exit data delete(A[:nx*ny], x[:ny], b[:nx])
delete[]A;
delete[]x;
delete[]b;


return 0;

}

