#include <iostream>
#include <memory>
using std::cout;
using std::endl;
using std::unique_ptr;

class Grid {
   public:
     Grid(int nv_) : nv( nv_ ) {}
     int get_nv() { return nv; }
     int nv;
};

class Sim {
   public:
     unique_ptr<Grid> grid;
     Sim(int n)   { grid = std::make_unique<Grid>(11);    }
     int get_nv() { return grid->get_nv(); }
 
};

int main () {
   Sim sim(11);
   cout << sim.get_nv() << endl;
}
