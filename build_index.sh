#!/bin/bash

if [ "$#" -lt 1 ]; then

	echo "Must specify the FASTA file"
	echo "Exiting"
	exit 1

fi

OPTIND=1 # Reset in case getopts has been used previously in the shell.

RATIO=""
PREFIX=""

# Initialize our own variables:

while getopts "r:p:" opt; do
    case "$opt" in

    r)  RATIO=$OPTARG
        ;;
    p)  PREFIX=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

FASTA=$@


if [ "$RATIO" = "" ]; then

	RATIO=16
fi

if [ "$PREFIX" = "" ]; then

	PREFIX=$FASTA

fi

sed -i "s,\#define OCC_INTV_SHIFT.*,\#define OCC_INTV_SHIFT 7,g" ./bwa_index/bwt.h 

echo "Compiling bwa"

make -C ./bwa_index clean
make -C ./bwa_index

echo "Building the suffix position array with compression ratio = $RATIO"

./bwa_index/bwa index -s sa -r $RATIO -p $PREFIX $FASTA

rm ${PREFIX}.bwt

echo "Building BWT array"

sed -i "s,\#define OCC_INTV_SHIFT.*,\#define OCC_INTV_SHIFT 6,g" ./bwa_index/bwt.h 

make -C ./bwa_index clean
make -C ./bwa_index

./bwa_index/bwa index -s bwt -p $PREFIX ${FASTA} 

rm ${PREFIX}.bwt1