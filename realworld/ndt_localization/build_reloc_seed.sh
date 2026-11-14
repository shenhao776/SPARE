#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/ndt_localization_build}
NDT=$HERE/ndt_omp_ros2
PREFIX=${PCL_PREFIX:-/usr}
CXX=${CXX:-g++}
INC="-I$PREFIX/include/pcl-1.12 -I$PREFIX/include/eigen3 -I$NDT/include"
LIBDIR=""; [ "$PREFIX" != "/usr" ] && LIBDIR="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib"
mkdir -p "$OUT/obj"
for f in voxel_grid_covariance_omp ndt_omp gicp_omp; do
  "$CXX" -DDISABLE_PCAP $INC -fopenmp -O3 -DNDEBUG -std=gnu++14 \
    -c "$NDT/src/pclomp/$f.cpp" -o "$OUT/obj/$f.o"
done
rm -f "$OUT/libndt_omp.a"
ar qc "$OUT/libndt_omp.a" "$OUT/obj/voxel_grid_covariance_omp.o" "$OUT/obj/ndt_omp.o" "$OUT/obj/gicp_omp.o"
ranlib "$OUT/libndt_omp.a"
"$CXX" -O2 -fopenmp -o "$OUT/reloc_seed" "$HERE/reloc_seed.cpp" $INC "$OUT/libndt_omp.a" $LIBDIR \
  -lpcl_io -lpcl_common -lpcl_filters -lpcl_kdtree -lpcl_search -lpcl_octree -lboost_system
echo "built: $OUT/reloc_seed"
