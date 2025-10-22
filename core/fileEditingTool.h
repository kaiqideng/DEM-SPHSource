#pragma once
#include <fstream>
#include <sstream>
#include <utility>
#include <string>
#include <tuple>
#include <cctype>
#include <iomanip>
#include <iostream>

//static void ltrim(std::string& s)
//{
//    size_t i = 0;
//    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
//    s.erase(0, i);
//}
//
//static bool getlineValid(std::ifstream& fin, std::string& line)
//{
//    while (std::getline(fin, line))
//    {
//        ltrim(line);
//        if (line.empty() || line[0] == '#') continue;
//        return true;
//    }
//    return false;
//}
//
//static int countValidNumbersStrict(const std::string& line)
//{
//    std::istringstream iss(line);
//    std::string token;
//    int count = 0;
//    while (iss >> token)
//    {
//        std::istringstream tokStream(token);
//        double d;
//        char c;
//        if (tokStream >> d && !(tokStream >> c))
//        {
//            ++count;
//        }
//    }
//    return count;
//}

#ifdef _WIN32
#include <io.h>
#include <direct.h>                 // _mkdir

#define MKDIR(path) _mkdir(path)    // returns 0 on success, -1 if already exists

static inline int removeVtuFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.vtu";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}

static inline int removeDatFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.dat";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}
#endif
