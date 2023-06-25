#!/bin/bash

neighbour_window_sizes=(0 0 0 2 2 2 4 4 4 4 0 0)

for attribute in "COS-INST-PHASE" "ENVELOPE" "INST-FREQ"
    do
        for i in {0..9..3}
            do
                x=${neighbour_window_sizes[$i]}
                y=${neighbour_window_sizes[$i+1]}
                z=${neighbour_window_sizes[$i+2]}
                echo "========================================================================"
                echo "Training model for x=$x, y=$y and z=$z and $attribute attribute..."
                
                if [[ -z $1 ]]; then
                    echo "Scheduler's address not found. Running local version."
                    docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model-hyperparam-search.py --attribute $attribute --data data/raw/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --output grid_search_models/$attribute-$x-$y-$z.json --save-pipeline-fig pipe-$x-$y-$z.png
                    echo "Local version ended."
                else
                    echo "Running multi-node version for scheduler on $1..."
                    docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model-hyperparam-search.py --attribute $attribute --data data/raw/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --address $1 --output grid_search_models/$attribute-$x-$y-$z.json --save-pipeline-fig pipe-$x-$y-$z.png
                    echo "Multi-node version ended."
                fi
                echo "Model trained."
                sleep 15
            done
    done