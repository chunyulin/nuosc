#pragma once
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <zlib.h>
using std::cout;
using std::endl;

// WriteLocal for MPI local output
class WriteLocal : public std::ofstream {
  public:
    template<typename T> friend WriteLocal& operator<<(WriteLocal&, T);
    WriteLocal() { };
    void init(const std::string& prefix, int rank) {
        // TODO: move profile log to random folder at local scratch and copy back after.
        char hname[20];
        gethostname(hname,sizeof(hname));
        std::string fname = prefix + "." + std::to_string(rank);
        this->open(fname.c_str(), std::ofstream::out | std::ofstream::trunc);
        *this << "## Logger for rank " << rank << " @ " << hname << endl;
    }

};

template<typename T>
inline WriteLocal& operator<<(WriteLocal& log, T op) {
    //write stream to the target file.
    auto& base_log = static_cast<std::ofstream&>(log);
    base_log << op;
    base_log.flush();
    return log;
}

// Wrapper for ASCII output 
class WriteText : public std::ofstream {
  public:
    template<typename T> friend WriteText& operator<<(WriteText&, T);
    WriteText() { };
    void init(const std::string& fname) {
        this->open(fname.c_str(), std::ofstream::out | std::ofstream::trunc);
    }

};
template<typename T>
inline WriteText& operator<<(WriteText& log, T op) {
    auto& base_log = static_cast<std::ofstream&>(log);
    base_log << op;
    base_log.flush();
    return log;
}


// Wrapper for binary output
class WriteBinary {
  gzFile fp;
  public:
    WriteBinary()  { };
    void init(std::string fname) {
      fp = gzopen(fname.c_str(),"wb");
    }
    ~WriteBinary() { gzclose(fp); };

    void write(char* data, size_t len) {
      gzwrite(fp, data, len);
    }
    void flush() {
      //gzflush(fp, Z_SYNC_FLUSH);
      gzflush(fp, Z_FULL_FLUSH);
    }
};


