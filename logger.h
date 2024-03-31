#pragma once
#include <fstream>
#include <iostream>
#include <unistd.h>

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

