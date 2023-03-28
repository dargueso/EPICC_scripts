#
# <envars_setup.sh>
#
# Define environment variables to set up WRF.
#

export LIBS_DIR=$HOME/WRFEnv/libraries
export WRF_DIR=$HOME/WRFEnv/WRFv4
export WPS_DIR=$HOME/WRFEnv/WPSv4
export CC=gcc
export CXX=g++
export F77=gfortran
export FC=gfortran
export F90=gfortran
export FFLAGS=-m64
export FCLAGS=-m64
export LD_LIBRARY_PATH=$LIBS_DIR/lib:$LD_LIBRARY_PATH
export PATH=$LIBS_DIR/bin:$PATH
export LDFLAGS=-L$LIBS_DIR/lib
export CPPFLAGS=-I$LIBS_DIR/include
export NETCDF=$LIBS_DIR
export HDF5=$LIBS_DIR
export JASPERLIB=$LIBS_DIR/lib
export JASPERINC=$LIBS_DIR/include
export J="-j 1"
export WRFIO_NCD_LARGE_FILE_SUPPORT=1

