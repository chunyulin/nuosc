#include "h5utils.h"

class H5ckpt {

 const int SDIM = 3;
 const int CDIM = 3;
 const int DIM0 = 8;
 const int DIM1 = 128;
 const int NV = 256;
 ulong N = DIM0*DIM1*NV;
 double* wdata;

public:
 H5ckpt() {

    wdata = new double[N];
    for (int i = 0; i < DIM0; i++)
        for (int j = 0; j < DIM1; j++) 
        for (int v = 0; v < NV; v++) {
            ulong idx = i*DIM1*NV+j*NV+v;
            wdata[idx] = idx;
     }

     cout << "Data size = " << (N*8.0/1024./1024.) << " MB.\n";
  }

 ~H5ckpt() {
    delete[] wdata;
 }

 void Checkpoint(int iter = 0) {

    std::string fname = "it" + std::to_string(iter) + ".h5";
    double *rdata = new double[N];

    hsize_t dims[3] = {DIM0, DIM1, NV};
    hsize_t chunk[3] = {1,4,NV};   // let CHUNK = 1024 = 128*8

    herr_t       status;

    unsigned int filter_info;
    hid_t avail = H5Zfilter_avail(H5Z_FILTER_SZIP);
    H5Zget_filter_info(H5Z_FILTER_SZIP, &filter_info);

    // Create H5 and global attribute
    hid_t h5fid = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    const hsize_t FIVE = 5;
    int nx[FIVE] = {1,2,3,4,5};

    addCharAttr(h5fid, "format", "COSEnu");
    addScalarAttr<double>(h5fid, "dx", 0.3);
    addScalarAttr<double>(h5fid, "alpha", 0.9);
    addIntArrAttr(h5fid, "nx", nx, &FIVE);

    // group for a iteration
    double phy_time = 1.20;
    std::string gtag = "/it" + std::to_string(iter);
    hid_t gid = H5Gcreate (h5fid, gtag.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    addScalarAttr<int>(gid, "iter", iter);
    addScalarAttr<double>(gid, "time", phy_time);

    // Create dset property for ZIP
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_szip(dcpl, H5_SZIP_NN_OPTION_MASK, 8);
    H5Pset_chunk(dcpl, CDIM, chunk);
    hid_t space = H5Screate_simple(SDIM, dims, NULL);

{
    // .. Create zipped dataset & write
    hid_t dset1 = H5Dcreate(gid, "ff::ee", H5T_STD_U32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(dset1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wdata);
    H5Dclose(dset1);
}
{
    // .. Create zipped dataset & write
    hid_t dset2 = H5Dcreate(gid, "ff::mm", H5T_STD_U32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(dset2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wdata);
    H5Dclose(dset2);
}

    // end of dumping iteration
    H5Gclose (gid);

    H5Fclose(h5fid);

    H5Sclose(space);
    H5Pclose(dcpl);
 }

 void RestartFromCheckpoint(int iter = 0) {

    std::string gtag = "/it" + std::to_string(iter);
    double *rdata = new double[N];

    std::string fname = "it" + std::to_string(iter) + ".h5";
    hid_t h5fid = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    char* format;
    double alpha;
    double dx;
    readCharAttr(h5fid, "format", &format); // HDF5 will allocate string memeoy. Need to free by HD5free()..
    readScalarAttr<double>(h5fid, "dx", &dx);
    readScalarAttr<double>(h5fid, "alpha", &alpha);

    cout << "Format: " << format << endl;
    cout << "Alpha: " << alpha << endl;
    cout << "dx: " << dx << endl;
    int *nx;
    readIntArrAttr(h5fid, "nx", &nx);
    cout << "N: " << nx[0] << " " << nx[1]  << " " << nx[2]  << " " << nx[3]  << " " << nx[4] << endl;

    // Read a iteration group
    auto gid = H5Gopen1(h5fid, gtag.c_str());

    int iter1;
    double time;
    readScalarAttr<int>(gid, "iter", &iter1);
    readScalarAttr<double>(gid, "time", &time);
    cout << " Iter: " << iter1 << endl;
    cout << " Time: " << time << endl;

{
    // read1
    auto dset = H5Dopen(gid, "ff::ee", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    H5Dclose(dset);
    for (int j = 0; j < 10; j++) {
            printf("%.1f ", rdata[1*DIM1+j]);
    }
    cout << endl;
}
{
    // read1
    auto dset = H5Dopen(gid, "ff::mm", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    H5Dclose(dset);

    for (int j = 0; j < 10; j++) {
            printf("%.1f ", rdata[1*DIM1+j]);
    }
    cout << endl;
}
    H5Fclose(h5fid);
    H5Gclose(gid);
 delete[] rdata;
delete[] nx;
 }

};


int main()
{

    H5ckpt ckpt;

    cout << "Writing ..." << endl;
    ckpt.Checkpoint();
    cout << "Reading ..." << endl;
    ckpt.RestartFromCheckpoint();
    return 0;
}
