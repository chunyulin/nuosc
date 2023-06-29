#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

#include <cufft.h>
//#include <cutil_inline.h>
//#include <shrQATest.h>
void fft(std::vector<double> x, std::vector<double> y);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE       1024
#define cutilSafeCall(x) x
#define cufftSafeCall(x) x
int main(int argc, char** argv)
{
    std::vector<double> x(SIGNAL_SIZE), y(SIGNAL_SIZE);
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        x[i] = rand() / (double)RAND_MAX;
        y[i] = 0;
    }
    
    fft(x,y);
  

}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void fft(std::vector<double> x, std::vector<double> y)
{
    uint SIZE = x.size();
    
    printf("[1DCUFFT] is starting...\n");

    cufftComplex* h_signal=(cufftComplex*)malloc(sizeof(cufftComplex) * SIZE);
    // Allocate host memory for the signal
    //Complex* h_signal = (Complex*)malloc(sizeof(Complex) * SIZE);
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < SIZE; ++i) {
        h_signal[i].x = x[i];
        h_signal[i].y = y[i];
    }

    int mem_size = sizeof(cufftComplex) * SIZE;

    // Allocate device memory for signal
    cufftComplex* d_signal;
    cutilSafeCall(cudaMalloc((void**)&d_signal, mem_size));

    // Copy host memory to device
    cutilSafeCall(cudaMemcpy(d_signal, h_signal, mem_size,
                              cudaMemcpyHostToDevice));



    // CUFFT plan
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, SIZE, CUFFT_C2C, 1));

    // Transform signal
    printf("Transforming signal cufftExecC2C\n");
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

    // Copy device memory to host
    cufftComplex* h_inverse_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * SIZE);
    cutilSafeCall(cudaMemcpy(h_inverse_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

    for(int i=0;i< SIZE;i++){
        h_inverse_signal[i].x= h_inverse_signal[i].x/(float)SIZE;
        h_inverse_signal[i].y= h_inverse_signal[i].y/(float)SIZE;

        printf("Residule : %f %f\n",h_signal[i].x-h_inverse_signal[i].x, h_signal[i].y-h_inverse_signal[i].y);
    }  



    //Destroy CUFFT context
    cufftSafeCall(cufftDestroy(plan));

    // cleanup memory
    free(h_signal);

    free(h_inverse_signal);
    cudaFree(d_signal);
}
