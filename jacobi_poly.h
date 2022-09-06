#ifndef JACOBI_POLY_H_
#define JACOBI_POLY_H_

#include<cmath>
#include<vector>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_eigen.h>

typedef double real;

// See implementation in Appendix A of the dG text book.
void JacobiP(int N, real a, real b, std::vector<real>&x, std::vector<real>&fx, int npoints);       // Jacobi polynomials
void GradJacobiP(int N, real a, real b, std::vector<real>&x, std::vector<real>&fx, int npoints);  // Derivatives of Jacobi polynomials 
void JacobiGQ(int N, real a, real b, std::vector<real>&x, std::vector<real>&w);                   // Gauss Quadratures
void JacobiGL(int N, real a, real b, std::vector<real>&x, std::vector<real>&w);                   // Gauss Lobatto Quadratures 
void Vandermonde1D(int N, std::vector<real>&x,  std::vector<std::vector<real>> &V, int npoints);                    // Vandermonde matrix
void GradVandermonde1D(int N, std::vector<real>&x, std::vector<std::vector<real>> &V, int npoints);                // Derivative of Vandermonde matrix
void Dmatrix1D(int N, std::vector<real>&x, std::vector<std::vector<real>> &V, std::vector<std::vector<real>> &D, int npoints);            // Derivative Matrix (u_h' = Du_h)
void InvMatrix(int N, std::vector<std::vector<real>> &V, std::vector<std::vector<real>> &Vinv);

void LSWInv(std::vector<real> &w, std::vector<std::vector<real>> &W_inv);

#endif
