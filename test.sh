#!/bin/bash
# PRL Project 2 test script: builds mm.cpp and runs it with m*k MPI processes,
# where m is the row count of A (first line of mat1.txt) and k is the column
# count of B (first line of mat2.txt).

set -e

m=$(head -n 1 mat1.txt | tr -d '[:space:]')
k=$(head -n 1 mat2.txt | tr -d '[:space:]')
np=$((m * k))

mpic++ -O2 -std=c++17 -o mm mm.cpp

mpirun --oversubscribe -np "$np" ./mm
