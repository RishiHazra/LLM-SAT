#!/bin/bash
# check if src/PyMiniSolvers/ exists
if [ ! -d "src/PyMiniSolvers/" ]; then
    cd src
    git clone https://github.com/liffiton/PyMiniSolvers.git
    cd PyMiniSolvers
    make
    cd ../..
fi

# check if src/d4/ exists
if [ ! -d "src/d4/" ]; then
    cd src
    git clone https://github.com/crillab/d4.git
    cd d4
    make
    cd ../..
fi

# check if src/sharpsat-td/ exists
# if [ ! -d "src/sharpsat-td/" ]; then
#     cd src
#     git clone https://github.com/Laakeri/sharpsat-td.git
#     cd sharpsat-td
#     ./setupdev.sh
#     cd ../..
# fi

# create directory for data
if [ ! -d "data/" ]; then
    mkdir data
fi
