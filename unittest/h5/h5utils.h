#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using std::cout;
using std::endl;

//
// Write/Read int array attribute
//
void addIntArrAttr(hid_t loc, const char* tag, int* data, const hsize_t * dim, int rank = 1) {
  auto ds = H5Screate(H5S_SIMPLE);
  H5Sset_extent_simple(ds, rank, dim, NULL);
  auto att = H5Acreate(loc, tag, H5T_NATIVE_INT, ds, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(att, H5T_NATIVE_INT, data);
  H5Sclose(ds);
  H5Aclose(att);
}
uint readIntArrAttr(hid_t loc, const char* tag, int** data) {
  auto att = H5Aopen(loc, tag, H5P_DEFAULT);
  auto aspace = H5Aget_space(att);
  auto rank   = H5Sget_simple_extent_ndims(aspace);
  uint npts = H5Sget_simple_extent_npoints(aspace);
  *data = new int[npts];
  auto atype  = H5Aget_type(att);
  H5Aread(att, atype, *data);
  H5Aclose(att);
  return npts;
}
//
// Write/Read c-cstring attribute
//
void addCharAttr(hid_t loc, const char* tag, const char* data) {
  hid_t attr_type = H5Tcopy(H5T_C_S1);
  H5Tset_size(attr_type, H5T_VARIABLE);
  hid_t space  = H5Screate(H5S_SCALAR);
  hid_t att = H5Acreate(loc, tag, attr_type, space, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(att, attr_type, &data);
  H5Sclose(space);
  H5Aclose(att);
}
void readCharAttr(hid_t loc, const char* tag, char** data) {
  auto att = H5Aopen(loc, tag, H5P_DEFAULT);
  int sz = H5Aget_storage_size(att);
  hid_t atype = H5Aget_type(att);
  H5Aread(att, atype, data);
  H5Aclose(att);
}

//
// Write/Read generic scalar attribute
//
template<typename T>
void addScalarAttr(hid_t loc, const char* tag, T data) { }
template<>
void addScalarAttr<int>(hid_t loc, const char* tag, int data) {
  hid_t ds = H5Screate(H5S_SCALAR);
  hid_t att = H5Acreate(loc, tag, H5T_NATIVE_INT, ds, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(att, H5T_NATIVE_INT, &data);
  H5Sclose(ds);
  H5Aclose(att);
}
template<>
void addScalarAttr<double>(hid_t loc, const char* tag, double data) {
  hid_t ds = H5Screate(H5S_SCALAR);
  hid_t att = H5Acreate(loc, tag, H5T_NATIVE_DOUBLE, ds, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(att, H5T_NATIVE_DOUBLE, &data);
  H5Sclose(ds);
  H5Aclose(att);
}
template<>
void addScalarAttr<float>(hid_t loc, const char* tag, float data) {
  hid_t ds = H5Screate(H5S_SCALAR);
  hid_t att = H5Acreate(loc, tag, H5T_NATIVE_FLOAT, ds, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(att, H5T_NATIVE_FLOAT, &data);
  H5Sclose(ds);
  H5Aclose(att);
}
template<typename T>
void readScalarAttr(hid_t loc, const char* tag, T* data) {
  auto att = H5Aopen(loc, tag, H5P_DEFAULT);
  hid_t atype = H5Aget_type(att);
  H5Aread(att, atype, data);
  H5Aclose(att);
}
