#include "solverBase.h"

void solverBase::solve()
{
    initialize();
    while (simPara.iStep <= simPara.numSteps)
    {
        update();
        simPara.iStep++;
    }
}

void solverBase::initialize()
{
    std::cout << "Initializing..." << std::endl;
    hos.contactModels = HostSolidContactModel(10);
    conditionInitialize();

    std::cout << "Using GPU Device " << gpuPara.deviceIndex << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(gpuPara.deviceIndex);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
        exit(1);
    }
	else
	{
		std::cout << "GPU ready." << std::endl;
	}

    simPara.numSteps = int((simPara.maximumTime) / simPara.timeStep) + 1;
    simPara.frameInterval = simPara.numSteps / simPara.numFrames;
    if (simPara.frameInterval < 1) simPara.frameInterval = 1;
    simPara.iStep = 0;
	simPara.iFrame = 0;

    buildDeviceData();
	std::cout << "Device data built" << std::endl;
    neighborSearch(dev, 0, 1, gpuPara.maxThreadsPerBlock);
    addBondData();

    removeVtuFiles(dir);
    removeDatFiles(dir);
    outputData();

	std::cout << "Initialization completed." << std::endl;
    simPara.iStep++;
}

void solverBase::update()
{
    neighborSearch(dev, simPara.iStep, fluidIntegrateGap, gpuPara.maxThreadsPerBlock);

    solidIntegrateBeforeContact(dev, simPara.timeStep, gpuPara.maxThreadsPerBlock);

    fluidIntegrate(dev, simPara.timeStep, simPara.iStep, fluidIntegrateGap, gpuPara.maxThreadsPerBlock);

    calSolidContactAfterFluidIntegrate(dev, simPara.timeStep, gpuPara.maxThreadsPerBlock);

    handleDataAfterCalculateContact();

    solidIntegrateAfterContact(dev, simPara.timeStep, gpuPara.maxThreadsPerBlock);

    if (simPara.iStep % simPara.frameInterval == 0)
    {
        simPara.iFrame++;
        std::cout << "Frame " << simPara.iFrame << " at time " << simPara.iStep * simPara.timeStep << std::endl;
        outputData();
    }
}

void solverBase::outputFluidVTU()
{
    if(simPara.iFrame > 0) dev.fluids.uploadState(hos.fluids);

    /* --- make sure "output" dir exists (ignore error if it does) --- */
    MKDIR(dir.c_str());

    /* --- build file name: output/fluid_XXXX.vtu --- */
    std::ostringstream fname;
    fname << dir << "/fluid_" << std::setw(4) << std::setfill('0') << simPara.iFrame << ".vtu";

    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());

    out << std::fixed << std::setprecision(10);      // full double precision

    /* ============ XML HEADER ============ */
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    /* ---- global FieldData: TIME + STEP ---- */
    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << simPara.iStep * simPara.timeStep << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << simPara.iStep << " </DataArray>\n"
        "    </FieldData>\n";

    int N = hos.fluids.points.num;
    /* ---- start Piece ---- */
    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    /* ---- Points ---- */
    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        const double3& p = hos.fluids.points.position[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    /* ---- Cells: one VTK_VERTEX per point ---- */
    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    /* ---- PointData ---- */
    out << "      <PointData Scalars=\"smooth length\">\n";

    /* helper lambdas replaced by small inline fns (C++03 safe) */
    {   /* scalar double array */
        out << "        <DataArray type=\"Float32\" Name=\"smooth length\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << hos.fluids.points.effectiveRadii[i];
        }
        out << "\n        </DataArray>\n";
    }
    /* vector<double3> helper */
    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , hos.fluids.dyn.velocities        }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < v.size(); ++i) {
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        }
        out << "\n        </DataArray>\n";
    }

    out << "        <DataArray type=\"Float32\" Name=\"density\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << ' ' << hos.fluids.rho0[i] + hos.fluids.dRho[i];
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        out << ' ' << hos.fluids.dRho[i] * pow(hos.fluids.c[i], 2);
    }
    out << "\n        </DataArray>\n";

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}

void solverBase::outputSolidVTU()
{
    if (simPara.iFrame > 0) dev.solids.uploadState(hos.solids);

    MKDIR(dir.c_str());
    std::ostringstream fname;
    fname << dir << "/solid_" << std::setw(4) << std::setfill('0') << simPara.iFrame << ".vtu";
    std::ofstream out(fname.str().c_str());
    if (!out) throw std::runtime_error("Cannot open " + fname.str());
    out << std::fixed << std::setprecision(10);
    const int N = hos.solids.points.num;
    out << "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <UnstructuredGrid>\n";

    out << "    <FieldData>\n"
        "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << simPara.iStep * simPara.timeStep << " </DataArray>\n"
        "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
        << simPara.iStep << " </DataArray>\n"
        "    </FieldData>\n";

    out << "    <Piece NumberOfPoints=\"" << N
        << "\" NumberOfCells=\"" << N << "\">\n";

    out << "      <Points>\n"
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) {
        const double3& p = hos.solids.points.position[i];
        out << ' ' << p.x << ' ' << p.y << ' ' << p.z;
    }
    out << "\n        </DataArray>\n"
        "      </Points>\n";

    out << "      <Cells>\n"
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int i = 1; i <= N; ++i) out << ' ' << i;
    out << "\n        </DataArray>\n"
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
    out << "\n        </DataArray>\n"
        "      </Cells>\n";

    out << "      <PointData Scalars=\"radius\">\n";

    out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << hos.solids.radius[i];
    out << "\n        </DataArray>\n";

	//out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
	//for (int i = 0; i < N; ++i) out << ' ' << hos.solids.pressure[i];
	//out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"clusterID\" format=\"ascii\">\n";
    for (int i = 0; i < N; ++i) out << ' ' << hos.solids.clusterID[i];
    out << "\n        </DataArray>\n";

	out << "        <DataArray type=\"Int32\" Name=\"clumpID\" format=\"ascii\">\n";
	for (int i = 0; i < N; ++i) out << ' ' << hos.solids.clumpID[i];
	out << "\n        </DataArray>\n";

    const struct {
        const char* name;
        const std::vector<double3>& vec;
    } vec3s[] = {
        { "velocity"       , hos.solids.dyn.velocities        },
		{ "angularVelocity", hos.solids.angularVelocities     },
		{ "normal"         , hos.solids.normal                }
    };
    for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
        out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double3>& v = vec3s[k].vec;
        for (size_t i = 0; i < v.size(); ++i)
            out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
        out << "\n        </DataArray>\n";
    }

    out << "      </PointData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n";
}


