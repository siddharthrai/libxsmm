#!/usr/bin/env bash

# args: iters N E M S P alpha

ITERS=1000
S=1
alpha=1.16
TH=56
n=2048
e=256
p=1

if [ "x$1" != "x" ] ; then ITERS=$1; fi
if [ "x$2" != "x" ] ; then S=$2; fi
if [ "x$3" != "x" ] ; then alpha=$3; fi
if [ "x$4" != "x" ] ; then TH=$4; fi
if [ "x$5" != "x" ] ; then n=$5; fi
if [ "x$6" != "x" ] ; then e=$6; fi
if [ "x$7" != "x" ] ; then p=$7; fi

export KMP_AFFINITY=granularity=fine,compact,1,0
BIN_FN=./main
NUMACTL_ARGS="numactl -m 0 "

if [ "${TH}" -gt "32" ]; then
  for T in ${TH} 32 16 8 4 1; do
    for N in ${n} ; do
      for E in ${e}; do
        for P in ${p}; do
          for M in 10000000 ; do
            export OMP_NUM_THREADS=$T
            echo "Running: $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha with $T threads"
            $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha
          done
        done
      done
    done
  done
else
  for T in 16 8 4 1; do
    for N in ${n} ; do
      for E in ${e}; do
        for P in ${p}; do
          for M in 10000000 ; do
            export OMP_NUM_THREADS=$T
            echo "Running: $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha with $T threads"
            $NUMACTL_ARGS $BIN_FN $ITERS $N $E $M $S $P $alpha
          done
        done
      done
    done
  done
fi

