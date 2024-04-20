#include <cstdlib>
#include <cstdio>
#include <mpi.h>
#include <openacc.h>
using std::endl;
using std::cout;

int main(int argc, char *argv[]) {

    MPI_Comm CartCOMM;

    int ranks = 1, myrank = 0;
    int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //int px = atoi(argv[1]);
    //int pz = atoi(argv[2]);
    

    // create Cartesian topology
    int periods[2] = {1,1};
    int pdims[2] = {0, 0};
    
    // Estimate the best rectangular domain decomposition
    MPI_Dims_create(ranks, 2, pdims);
    int px = pdims[0];
    int pz = pdims[1];

    if (!myrank) cout << "Creating CartGrid ......";
    MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 0, &CartCOMM);

    // calcute local geometry from computational domain (x0_, x1_, z0_, z1_)
    int coords[2];
    MPI_Cart_coords(CartCOMM, myrank, 2, coords);
    int rx = coords[0];
    int rz = coords[1];

    int lowerX, upperX, lowerZ, upperZ;
    MPI_Cart_shift(CartCOMM, 0, 1, &lowerX, &upperX);
    MPI_Cart_shift(CartCOMM, 1, 1, &lowerZ, &upperZ);

    // 
    // Get shared comm
    // 
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_size(shmcomm, &shmsize);
    MPI_Comm_rank(shmcomm, &shmrank);
    
    int ngpus;
    
    // OpenACC
    #ifdef _OPENACC
    if (myrank==0) printf("\n\nOpenACC Enabled.\n" );
    auto dev_type = acc_get_device_type();
    ngpus = acc_get_num_devices( dev_type );
    acc_set_device_num( shmrank, dev_type );
    #endif


    printf("[ Rank %2d ]  SHem %d / %d  My coor:( %d %d )  nbX:( %d %d ) nbZ:( %d %d ) GPU: %d / %d\n",
          myrank, shmrank, shmsize, rx, rz, lowerX, upperX, lowerZ, upperZ, shmrank, ngpus);

    const int S = 10;
    int N = ranks * S;
    double *a = new double[N];

    #pragma acc parallel loop independent
    for(int i=0;i<S;++i) {
       a[myrank*S+i] = myrank;
    }

    double sum = 0;
    #pragma acc parallel loop reduction(+:sum)
    for(int i=0;i<S;++i) {
       sum += a[myrank*S+i];
    }

    
    cout << "Local sum from " << myrank << " : " << sum << endl;
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (!myrank) cout << "Total sum : " << sum << endl;

    MPI_Finalize();    // error because this is called before the deconstructor of CartGrid
    
    delete[] a;
    return 0;
}
