#!/bin/bash

eval "$(conda shell.bash hook)"

set -e

# set the ENV_PATH variable to an absolute path of your conda environment directory
ENV_PATH=# enter your path here

START_PWD=$(pwd)

conda activate $ENV_PATH
conda install -c conda-forge conan -y
conda install boost -y

git clone https://github.com/facebookresearch/StarSpace.git

cd StarSpace/python

# set boost dir in makefiles
sed -i "s/^BOOST_DIR.*$/BOOST_DIR = $(echo $ENV_PATH | sed -e 's/[\/&]/\\&/g')\/include\//" ../makefile_py
# and proper variables in CMakeLists.txt
sed -i "s/$(echo /usr/local/bin/boost_1_63_0/ | sed -e 's/[\/&]/\\&/g')/$(echo $ENV_PATH | sed -e 's/[\/&]/\\&/g')\/include\//" CMakeLists.txt
sed -i "1s/^/set(PYTHON_LIBRARY \"$(echo $ENV_PATH | sed -e 's/[\/&]/\\&/g')\/lib\")\n/" CMakeLists.txt
sed -i "1s/^/set(PYTHON_INCLUDE_DIR \"$(echo $ENV_PATH | sed -e 's/[\/&]/\\&/g')\/include\/python3.7m\")\n/" CMakeLists.txt

chmod +x build.sh
./build.sh

cp test/starwrap.so $ENV_PATH/lib/python3.7/site-packages/

cd $START_PWD
rm -rf StarSpace

conda deactivate
