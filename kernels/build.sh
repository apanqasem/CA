#!/bin/bash

function usage() {
/bin/cat 2>&1 <<"EOF"
     
    Build kernels with CA layout 

    For more details, see paper.  

    Usage: build.sh [ options ] 
    
    Options without values:
       --help             print this help message
       -v,--verbose       print debug info during kernel execution 

    Options with values:
       -m,--mode <action>          Action. <action> = build or clean 
       -l,--layout <layout>        Data layout. <layout>=AOS, DA or CA. Default=AOS

    Options with values:

    Examples:
       ./build.sh -p 2dconv -l CA   // build 2Dconv with CA layout 


EOF
}

if [ $# -lt 1 ] || [ "$1" = "--help" ]; then
	usage
  exit 0
fi


while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -v|--verbose)
      verbose="VERBOSE"
			;;
    -p|--prog)
      prog="$2"
			shift
      ;;
		-m|--mode)
			mode="$2"
			shift
			;;
		-t|--tile)
			tile="$2"
			shift
			;;
    -l|--layout)
      layout="$2"
      shift
			;;
    -a|--alloc)
      alloc="$2"
			shift
			;;
    -c|--convert)
      convert="CONVERT"
			;;
		--mem)
			mem="$2"
			shift
			;;
		-i|--intensity)
			intensity="$2"
			shift
			;;
		-s|--size)
			size="$2"
			shift
			;;
		-p|--sparsity)
			sparsity="$2"
			shift
			;;
		-u|--unroll)
			unroll="$2"
			shift
			;;
		--sweeps)
			sweeps="$2"
			shift
			;;
		-d|--coarsen)
			coarsen="$2"
			shift
			;;
    *)
			echo "Unknown option:" $key
			exit 0
			;;
  esac
  shift
done

[ "$mode" ] || { mode="build";} 
[ "$prog" ] || { prog="2dconv";} 
[ "$layout" ] || { layout="DEFAULT"; }
[ "$tile" ] || { tile="4";} 


EXEC=$prog
SRC=$prog.cu
PROGDIR=${prog}

cd ${PROGDIR}

CA_DIR="../../ca"
CA_OBJ=${CA_DIR}/ca.o
UTIL_OBJ=${CA_DIR}/util.o

OBJS="${CA_OBJ} ${UTIL_OBJ}"
POLY_INC_DIR=../inc

INCPATH="-I${CA_DIR} -I${POLY_INC_DIR}"

PARAM_DEFS="-D${agent} -D${alloc} -D${convert} -D${layout} -D${verbose} -DMEM${mem} -DIMGS=${size}\
            -DSPARSITY_VAL=${sparsity} -DUNROLL_VAL=${unroll} -DSWEEP_VAL=${sweeps} -DTILESIZE=${tile}\
            -DCOARSENFACTOR=${coarsen} -D${module_type}"

PARAM_DEFS="-D${layout} -DTILESIZE=${tile}"

if [ $mode = "clean" ]; then
	rm -f *~ *.o ${EXEC} 
fi

if [ $mode = "build" ]; then
	echo "g++ -I ${CA_DIR} -c ${PARAM_DEFS} ${CA_DIR}/ca.c"
	g++ -I ${CA_DIR} -c ${PARAM_DEFS} ${CA_DIR}/ca.c
	echo "nvcc -O3 -arch sm_30 -ccbin g++ ${PARAM_DEFS} ${INCPATH} ${SRC} ${OBJS} -o ${EXEC}"
	nvcc -O3 -arch sm_30 -ccbin g++ ${PARAM_DEFS} ${INCPATH} ${SRC} ${OBJS} -o ${EXEC}
fi

cd ../
