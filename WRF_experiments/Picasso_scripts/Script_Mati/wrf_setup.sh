#!/bin/sh
#
# <wrf_installation_libraries.sh>
#

set -xe
# Uncomment (and modify) if module loading is required.
module load openmpi_gcc/3.1.6_gcc7.5.0
. ./envars_setup.sh
mkdir -p $LIBS_DIR
mkdir -p $LIBS_DIR/src

#-------------------- Install libraries --------------------#
# zlib
cd $LIBS_DIR
wget https://zlib.net/fossils/zlib-1.2.11.tar.gz 
tar xvf zlib-1.2.11.tar.gz
cd zlib-1.2.11/

./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib
make check 2>&1 | tee make_check.log
make install

cd ..
rm zlib-1.2.11.tar.gz
mv zlib-1.2.11 src

## libpng
cd $LIBS_DIR
wget https://sourceforge.net/projects/libpng/files/libpng16/1.6.37/libpng-1.6.37.tar.gz 
tar xvf libpng-1.6.37.tar.gz
cd libpng-1.6.37

./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib
make check 2>&1 | tee make_check.log
make install

cd ..
rm libpng-1.6.37.tar.gz
mv libpng-1.6.37 src

# HDF5
cd $LIBS_DIR
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.1/src/hdf5-1.10.1.tar.gz
gzip -d hdf5-1.10.1.tar.gz
tar xf hdf5-1.10.1.tar
cd hdf5-1.10.1

./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib --with-zlib=$LIBS_DIR --enable-fortran --enable-hl

make check 2>&1 | tee make_check.log
make install

cd ..
rm hdf5-1.10.1.tar
mv hdf5-1.10.1 src

cd $LIBS_DIR
wget https://github.com/curl/curl/releases/download/curl-7_58_0/curl-7.58.0.tar.xz
tar xvf curl-7.58.0.tar.xz
cd curl-7.58.0

./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib
make 
make install

cd ..
rm curl-7.58.0.tar.xz
mv curl-7.58.0 src

## netCDF (c version)
##
## Test 67 fails in make check. Nothing to worry about, fixed by adding
## the option --disable-dap-remote-tests in the configure step.
## Further reading:
##   https://github.com/Unidata/netcdf-c/issues/2242

cd $LIBS_DIR
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.6.3.tar.gz 
tar xvf v4.6.3.tar.gz
cd netcdf-c-4.6.3

./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib --disable-dap-remote-tests
make check 2>&1 | tee make_check.log
make install

cd ..
rm v4.6.3.tar.gz 
mv netcdf-c-4.6.3 src

# netCDF (fortran version)
cd $LIBS_DIR
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.4.5.tar.gz
tar xvf v4.4.5.tar.gz
cd netcdf-fortran-4.4.5
./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib
make check 2>&1 | tee make_check.log
make install

cd ..
rm v4.4.5.tar.gz
mv netcdf-fortran-4.4.5 src

# JasPer
## https://github.com/NCAR/ncl/issues/130
cd $LIBS_DIR
wget https://ece.engr.uvic.ca/~frodo/jasper/software/jasper-1.900.29.tar.gz 
tar xvf jasper-1.900.29.tar.gz

cd jasper-1.900.29
autoreconf -i
./configure --prefix=$LIBS_DIR --libdir=$LIBS_DIR/lib
make
make install

cd ..
rm jasper-1.900.29.tar.gz
mv jasper-1.900.29 src

##-------------------- Install WRF --------------------#

# sm+dm not recommended: https://forum.mmm.ucar.edu/phpBB3/viewtopic.php?t=8970#p16242
# landread issue: https://github.com/wrf-model/WRF/pull/1072/commits/c64512424c5d1741ab290897f473d31edc264966

cd $LIBS_DIR/..
wget https://github.com/wrf-model/WRF/archive/refs/tags/v4.3.3.tar.gz
tar xzvf v4.3.3.tar.gz
mv WRF-4.3.3 WRFv4
rm v4.3.3.tar.gz

cd $WRF_DIR
./configure
./compile em_real 2>&1 | tee compile.log

#-------------------- Install WPS --------------------#

# Fix for version 3.9.1.:https://github.com/wrf-model/WPS/pull/50
# https://github.com/wrf-model/WPS/pull/119


cd $LIBS_DIR/..
wget https://github.com/wrf-model/WPS/archive/refs/tags/v4.3.1.tar.gz
tar xzvf v4.3.1.tar.gz
mv WPS-4.3.1 WPSv4
rm v4.3.1.tar.gz

cd $WPS_DIR
./configure
export MPI_LIB=""
./compile 2>&1 | tee compile.log

