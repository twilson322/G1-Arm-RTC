#!/usr/bin/env bash
# Full pipeline: scene, build, tests, deterministic experiments, real-time runs, benchmarks, plots.
# About 15 minutes on 2 cores, most of it the collision-detection sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d models/assets ] || ./tools/fetch_assets.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
(cd build && ctest --output-on-failure)
mkdir -p results
./build/g1rtc_experiments all results | tee results/experiments.txt
./build/g1rtc_rt --tag nominal --out results | tee results/rt_nominal.txt
./build/g1rtc_rt --tag stress2 --stress 2 --out results | tee results/rt_stress2.txt
if [ -x build/g1rtc_bench ]; then
  ./build/g1rtc_bench --benchmark_out=results/bench.json --benchmark_out_format=json | tee results/bench.txt
fi
python3 scripts/plot.py
