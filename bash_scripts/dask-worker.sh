#!/bin/bash

if [[ -z $1 ]]; then
    echo "You need to provide the scheduler's address!"
    exit 1;
else
    docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu dask worker $1
fi
