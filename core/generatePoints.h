#pragma once
#include <vector>
#include "myVec.h"

inline std::vector<double3> getRegularPackedPoints(double3 origin, double3 size, double spacing)
{
    int numPX = int((size.x + 1.e-10) / spacing);
    int numPY = int((size.y + 1.e-10) / spacing);
    int numPZ = int((size.z + 1.e-10) / spacing);
    if (numPX == 0 || numPY == 0 || numPZ == 0)
    {
        std::vector<double3> p(1, origin + 0.5 * size);
        return p;
    }
    double dx = size.x / double(numPX);
    double dy = size.y / double(numPY);
    double dz = size.z / double(numPZ);
    std::vector<double3> positions;
    for (int x = 1; x <= numPX; x++)
    {
        for (int y = 1; y <= numPY; y++)
        {
            for (int z = 1; z <= numPZ; z++)
            {
                double3 pos = make_double3(0, 0, 0);
                pos.x = origin.x + dx * (double(x) - 0.5);
                pos.y = origin.y + dy * (double(y) - 0.5);
                pos.z = origin.z + dz * (double(z) - 0.5);
                positions.push_back(pos);
            }
        }
    }
    return positions;
}

inline std::vector<double3> getHexagonalClosePackedPoints(double3 origin, double3 size, double spacing)
{
    int numPX = int(size.x / spacing);
    int numPY = int(size.y / sqrt(3.0) / spacing * 2.0) - 1;
    int numPZ = int(size.z / sqrt(6.0) / spacing * 3.0);
    double dx = spacing;
    double dy = spacing * sqrt(3.0) / 2.0;
    double dz = spacing * sqrt(6.0) / 3.0;
    double3 offset = make_double3(0, 0, 0);
    double3 realSize = make_double3(0, 0, 0);
    realSize.x = double(2. * numPX - 1 + 2) * dx / 2. + 0.5 * spacing;
    realSize.y = double(2. * numPY - 1 + 2. / 3.) * dy / 2. + (1. / sqrt(3.) - 0.5) * dy + 0.5 * spacing;
    realSize.z = 0.5 * (spacing - dz) + (2. * numPZ - 1) * dz / 2. + 0.5 * spacing;
    offset.x = (size.x - realSize.x) / 2.;
    offset.y = (size.y - realSize.y) / 2.;
    offset.z = (size.z - realSize.z) / 2.;
    std::vector<double3> positions;
    for (int x = 1; x <= numPX; x++)
    {
        for (int y = 1; y <= numPY; y++)
        {
            for (int z = 1; z <= numPZ; z++)
            {
                double3 pos = make_double3(0, 0, 0);
                pos.x = origin.x + offset.x + double(2. * x - 1 + y % 2 + z % 2) * dx / 2.;
                pos.y = origin.y + offset.y + double(2. * y - 1 + 2. * (z % 2) / 3.) * dy / 2. + (1. / sqrt(3.) - 0.5) * dy;
                pos.z = origin.z + offset.z + 0.5 * (spacing - dz) + (2. * z - 1) * dz / 2.;
                positions.push_back(pos);
            }
        }
    }
    return positions;
}