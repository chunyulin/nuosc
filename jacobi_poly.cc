//  Originate by https://github.com/bernatguillen/dG_project

#include<cassert>
#include "jacobi_poly.h"
#include<gsl/gsl_sort.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_cblas.h>
#include<gsl/gsl_sf_gamma.h>

//
//  P^(a=0,b=0)_n(x) reduced to Legendre polynominals.
//
void JacobiP(int P, real a, real b, std::vector<real> &x, std::vector<real> &fx, int npoints){
	const real Gamma0 = pow(2.0,a+b+1)/(a+b+1)*gsl_sf_gamma(a+1)*gsl_sf_gamma(b+1)/gsl_sf_gamma(a+b+1);
	real **PL;
	PL = new real*[P+1];
	for(int n = 0; n<P+1; ++n){
		PL[n] = new real[npoints];
	}
	//Initial values P=0, 1
	for(int i = 0; i<npoints; ++i){
		PL[0][i] = 1.0/sqrt(Gamma0);
	}
	if(P==0){
		for(int i = 0; i<npoints; ++i){
			fx[i] = PL[0][i];
		}
		return;
	}  
	real Gamma1 = Gamma0*(a+1)*(b+1)/(a+b+3);
	for(int i = 0; i<npoints; ++i){
		PL[1][i] = ((a+b+2.0)*x[i]*0.5+(a-b)*0.5)/sqrt(Gamma1);
	}
	if(P==1){
		for(int i = 0; i<npoints; ++i){
			fx[i] = PL[1][i];
		}
		return;
	}
	real aold = 2.0/(2+a+b)*sqrt((a+1)*(b+1)/(a+b+3));
	real anew,h1,bnew;
	for(int n = 1; n<=P-1; ++n){
		h1 = 2*n+a+b;
		anew = 2/(h1+2)*sqrt((n+1)*(n+1+a+b)*(n+1+a)*(n+1+b)/(h1+1)/(h1+3));
		bnew = - (a*a - b*b)/h1/(h1+2);
		for(int i = 0; i<npoints; ++i){
			PL[n+1][i] = 1.0/anew*(-aold*PL[n-1][i]+(x[i]-bnew)*PL[n][i]);
		}
		aold = anew;
	}
	for(int i = 0; i<npoints; ++i){
		fx[i] = PL[P][i];
	}
	for(int n = 0; n<P+1; ++n){
		delete [] PL[n];
	}
	delete [] PL;
	return;
}

// Compute P'th order Gauss quadrature points, x, and weights, w, associated with JacobiP(a,b,P)
void JacobiGQ(int P, real a, real b, std::vector<real> &x, std::vector<real> &w) {

	if (P==0) {
		x[0] = (a-b)/(a+b+2);
		w[0] = 2;
		return;
	}

	gsl_matrix *T = gsl_matrix_calloc(P+1,P+1);
	for(int i = 0; i<P+1; i++)  gsl_matrix_set(T,i,i,-(a*a-b*b)/((2*i+a+b)*(2*i+a+b+2)));

	if(a+b <= 1e-15) gsl_matrix_set(T,0,0,0);
	for (int i = 1; i <P+1; i++) {
		gsl_matrix_set(T, i  , i-1, 2.0/(2*i+a+b)*sqrt(i*(i+a+b)*(i+a)*(i+b)/((2*i+a+b-1)*(2*i+a+b+1))));
		gsl_matrix_set(T, i-1,   i, gsl_matrix_get(T,i,i-1));
	}
	gsl_vector *eigval = gsl_vector_alloc(P+1);
	gsl_matrix *eigvec = gsl_matrix_alloc(P+1,P+1);
	gsl_eigen_symmv_workspace *wrk = gsl_eigen_symmv_alloc(P+1);
	gsl_eigen_symmv(T,eigval,eigvec,wrk);
	gsl_eigen_symmv_free(wrk);
	for (int i = 0; i < P+1; ++i){
		x[i] = gsl_vector_get(eigval,i);
		w[i] = pow(gsl_matrix_get(eigvec,0,i),2)*pow(2,a+b+1)/(a+b+1)*gsl_sf_gamma(a+1)*gsl_sf_gamma(b+1)/gsl_sf_gamma(a+b+1);
	}
	gsl_sort2(x.data(),1,w.data(),1,P+1);
	gsl_vector_free(eigval);
	gsl_matrix_free(eigvec);
	gsl_matrix_free(T);

}

void JacobiGL(int P, real a, real b, std::vector<real> &x, std::vector<real> &w){
	if(P==1){
		x[0] = -1.0;
		x[1] = 1.0;
		w[0] = w[1] = 1.0;
		return;
	}
	JacobiGQ(P-2,a+1,b+1,x,w);
	for (int i = P-1; i > 0; --i){
		x[i] = x[i-1];
	}
	x[0] = -1.0;
	x[P] = 1.0;
	JacobiP(P, 0, 0, x, w, P+1);  // normalized Legendre poly
	std::cout << "w ==============: ";
	for (int i = 0; i <= P; ++i){
		w[i] = (2*P+1)/(P*(P+1)*w[i]*w[i]);
	}
	std::cout << std::endl;

	//if (P%2==0)  x[P/2] = 0.0;
	return;
}

void InvMatrix(int P, std::vector<std::vector<double>> &M, std::vector<std::vector<double>> &M_inv){
	gsl_matrix *M_gsl = gsl_matrix_alloc(P,P);
	gsl_matrix *M_gsl_inv = gsl_matrix_alloc(P,P);
	gsl_permutation *perm = gsl_permutation_alloc(P);
	for(int i = 0; i<P; ++i){
		for(int j = 0; j<P; ++j){
			gsl_matrix_set(M_gsl, i,j, M[i][j]);
		}
	}
	int s;
	gsl_linalg_LU_decomp(M_gsl, perm, &s);
	gsl_linalg_LU_invert(M_gsl, perm, M_gsl_inv);
	for(int i =0; i<P; ++i){
		for(int j =0; j<P; ++j){
			M_inv[i][j] = gsl_matrix_get(M_gsl_inv, i,j);
		}
	}
	gsl_matrix_free(M_gsl_inv);
	gsl_matrix_free(M_gsl);
	gsl_permutation_free(perm);
	return;
}

void Vandermonde1D(int P, std::vector<real> &x, std::vector<std::vector<double>> &V, int npoints){

	std::vector<real> fx(P+1);

	for (int i = 0; i < npoints; ++i){
		for (int j = 0; j < P+1; ++j){
			V[i][j] = 0;
		}
	}
	for(int j = 0; j<P+1; ++j){
		JacobiP(j,0,0,x,fx,npoints);
		for(int i = 0; i<npoints; ++i){
			V[i][j] = fx[i];   // V_{ij} = P_j(x_i)
		}
	}
	return;
}

void GradJacobiP(int P, real a, real b, std::vector<real>&x, std::vector<real>&fx, int npoints){
	//Derivative of JacobiP
	if(P==0){
		for(int i = 0; i<npoints; ++i){
			fx[i] = 0.0;
		}
	}else{
		JacobiP(P-1,a+1,b+1,x,fx,npoints);
		for(int i = 0; i<npoints; ++i){
			fx[i] *= sqrt(P*(P+a+b+1));
		}
	}
	return;
}

void GradVandermonde1D(int P, std::vector<real> &x, std::vector<std::vector<real>> &V, int npoints){
	std::vector<real> fx(P+1);
	for (int i = 0; i < npoints; ++i){
		for (int j = 0; j < P+1; ++j){
			V[i][j] = 0;
		}
	}
	for(int j = 0; j<P+1; ++j){
		GradJacobiP(j,0,0,x,fx,npoints);
		for(int i = 0; i<npoints; ++i){
			V[i][j] = fx[i];   //  V_{ij}' = (DV)_{ij}
		}
	}
	return;
}

void Dmatrix1D(int P, std::vector<real> &x, std::vector<std::vector<real>> &V, std::vector<std::vector<real>> &D, int np){

	assert(np==P+1);

	std::vector<std::vector<real>> Vr { static_cast<size_t>(np), std::vector<real>(static_cast<size_t>(np), 0.0) };

	GradVandermonde1D(P, x, Vr, np);  //  Vr := V_{ij}' = (DV)_{ij}
	gsl_matrix *V_gsl = gsl_matrix_alloc(np,np);
	gsl_matrix *V_inv = gsl_matrix_alloc(np,np);
	gsl_permutation *perm = gsl_permutation_alloc(np);
	for(int i = 0; i<np; ++i)
	for(int j = 0; j<np; ++j) {
	    gsl_matrix_set(V_gsl, i,j, V[i][j]);
	}
	int s;
	gsl_linalg_LU_decomp(V_gsl, perm, &s);
	gsl_linalg_LU_invert(V_gsl, perm, V_inv);
	for(int i = 0; i<np; ++i)
	for(int j = 0; j<np; ++j) {
	    D[i][j] = 0;
	    for(int k = 0; k<np; ++k) {
		D[i][j] += Vr[i][k]*gsl_matrix_get(V_inv,k,j);
	    }
	}
	gsl_matrix_free(V_inv);
	gsl_matrix_free(V_gsl);
	gsl_permutation_free(perm);
	return;
}

