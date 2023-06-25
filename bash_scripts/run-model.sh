#!/bin/bash

neighbour_window_sizes=(0 0 0 2 2 2 4 4 4 4 0 0)

for i in 1
    do
        x=${neighbour_window_sizes[$i]}
        y=${neighbour_window_sizes[$i+1]}
        z=${neighbour_window_sizes[$i+2]}
        echo "========================================================================"
        
        if [[ -z $1 ]]; then
            echo "Scheduler's address not found. Running local version."
            docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 run-model.py --ml-model models/CIP-ml-model-$x-$y-$z.json --data data/raw/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --output CIP-ml-model-$x-$y-$z.npy
            echo "Local version ended."
        else
            echo "Running multi-node version for scheduler on $1..."
            docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 run-model.py --ml-model models/CIP-ml-model-$x-$y-$z.json --data data/raw/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --address $1 --output CIP-ml-model-$x-$y-$z.npy
            echo "Multi-node version ended."
        fi
        echo "Model ran successfully."
        sleep 15
    done