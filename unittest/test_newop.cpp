#include <iostream>
using std::cout;
using std::endl;

class Grid {
   public:
     Grid(int nv_) : nv( nv_ ) {}
     int get_nv() { return nv; }
     int nv;
};

class Sim {
   public:
     Grid *grid;
     Sim(int nv_)  { grid = new Grid(nv_); }
     ~Sim()    {  delete grid; }
};

int main () {
   Sim sim(10);
   cout << sim.grid->get_nv() << endl;
}
