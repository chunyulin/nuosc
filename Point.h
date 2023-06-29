#pragma once
template <typename T>
class PointT
{
    public:
        PointT(): x(0), y(0), z(0) {}
        explicit PointT(const T& x_, const T& y_, const T& z_) :  x(x_),y(y_),z(z_) {}

        friend PointT operator+(PointT a, const PointT& b) {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            return a;
        }
        friend PointT operator-(PointT a, const PointT& b) {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
            return a;
        }
        friend PointT operator*(PointT a, const T k) {
            a.x = k * a.x;
            a.y = k * a.y;
            a.z = k * a.z;
            return a;
        }
        friend PointT operator*(const T k, PointT a) {
            return a * k;
        }
        friend T operator|(const PointT& a, const PointT& b) {
            return a.x*b.x+a.y*b.y+a.z*b.z;
        }
        friend PointT operator*(const PointT& a, const PointT& b) {
            return PointT( a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x  );
        }
        friend std::ostream& operator<<(std::ostream& os, const PointT& a) {
            os << a.x << " " << a.y << " " << a.z;
            return os;
        }
        // copy assignment (copy-and-swap idiom)
        PointT& operator=(PointT a) noexcept  {
            x = a.x;
            y = a.y;
            z = a.z;
            return *this;
        }

        inline T norm2() {
            return (x*x+y*y+z*z);
        }
        inline T norm() {
            return std::sqrt(x*x+y*y+z*z);
        }

        T x,y,z;
};

typedef PointT<double> Point;

double det(const Point& A, const Point& B, const Point& C) {
    return A|(B*C);
}

Point circumcenter(const Point& A, const Point& B, const Point& C) {
    auto ABxAC = (B-A)*(C-A);
    auto AC = C-A;
    auto AB = B-A;
    return A + ( 0.5 / ABxAC.norm2() ) * (ABxAC * ( AC.norm2()*AB - AB.norm2()*AC ));
}
