#include <mpi.h>

test_reduce(int argc, char *argv[]) {


    int lowerZ = 0, upperZ = 0, lowerX = 0, upperX = 0;
        MPI_Comm CartCOMM;
int px=1, py=1;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

            int periods[2] = {1,1};
            int pdims[2] = {px, pz};
            MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 0, &CartCOMM);

            int coords[2];
            MPI_Cart_coords(CartCOMM, myrank, 2, coords);
            rx = coords[0];
            rz = coords[1];

            MPI_Cart_shift(CartCOMM, 0, 1, &lowerX, &upperX);
            MPI_Cart_shift(CartCOMM, 1, 1, &lowerZ, &upperZ);
#endif


           real buf[] = {0,0,0,0};

            MPI_Request reqs[2];
                MPI_Isend(&buf[0],2, MPI_DOUBLE, upperX, 9, CartCOMM, &reqs[0]);
                MPI_Irecv(&buf[2],2, MPI_DOUBLE, lowerX, 9, CartCOMM, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
           cout << "=== " << buf[2] << " " << buf[3] << endl;



MPI_Finalize();
}




int main(int argc, char *argv[]) {

test_reduce(argc, argv);


return 0;

}

