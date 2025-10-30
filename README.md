# DEM-SPHSource
DEM-SPH GPU acceleration
Repository layout
DEM-SPHSource/
├─ core/            # Library code (.cpp/.cu/.h)
├─ driver/cases/    # Entry points with main() (e.g., damBreak.cpp)
├─ CMakeLists.txt
├─ README.md
└─ animation.mp4    # Demo (optional)

Requirements
NVIDIA Driver compatible with your CUDA Toolkit
CUDA Toolkit ≥ 11.8 (validated on 12.4 / 12.9)
CMake ≥ 3.18
GCC 9–12 (match your CUDA version)
Build system: Ninja (recommended) or GNU Make
Optional: ParaView (visualization), Nsight Systems/Compute (profiling)

GPU arch hints for -DCUDA_ARCHS:
T4=75, A100=80, RTX A4000=86, RTX 4090/L40S=89, H100=90

Quick start (Linux)
1) Install tools (Ubuntu example)
sudo apt update
sudo apt install -y build-essential cmake ninja-build git
nvidia-smi
nvcc --version

2) Configure & build (Ninja)
SRC=.../DEM-SPH/DEM-SPHSource
BLD=.../DEM-SPH/build
NVCC=/usr/local/cuda/bin/nvcc
CUDA_ROOT=$(dirname "$NVCC")/..

rm -rf "$BLD"
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/tmp CUDACXX="$NVCC" \
/usr/bin/cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCUDA_ARCHS="86"

env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/tmp \
ninja -C "$BLD" -j"$(nproc)"

No Ninja?

cmake -S "$SRC" -B "$BLD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCUDA_ARCHS="86"
make -C "$BLD" -j"$(nproc)"

3) Run
"$BLD"/bin/damBreak --help
cd "$SRC"
"$BLD"/bin/damBreak


Select GPUs:

CUDA_VISIBLE_DEVICES=0 "$BLD"/bin/damBreak

Build options

-DCUDA_ARCHS="86" target SMs (use semicolons for fat binaries, e.g. "80;86;89")

-DUSE_FAST_MATH=ON enables --use_fast_math for CUDA (if exposed in CMake)

Device linking is enabled via CUDA_SEPARABLE_COMPILATION ON

Example:

cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUDA_ARCHS="86" \
  -DUSE_FAST_MATH=ON

Visualization (ParaView)

Official binary:

# download ParaView-*-Linux-Python*.tar.xz to ~/Downloads
cd ~ && mkdir -p apps && cd apps
tar -xf ~/Downloads/ParaView-*-Linux-Python*.tar.xz
./ParaView-*/bin/paraview
