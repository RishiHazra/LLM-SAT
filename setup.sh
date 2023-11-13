#!/bin/bash
mkdir src
cd src
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
cd ../..

mkdir data