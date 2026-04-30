#ifndef POSE2D_H
#define POSE2D_H

struct Pose2d {
    double x_, y_, theta_;
    
    Pose2d() : x_(0.0), y_(0.0), theta_(0.0) {}
    Pose2d(double x, double y, double theta) : x_(x), y_(y), theta_(theta) {}
};

#endif // POSE2D_H

