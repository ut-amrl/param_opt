# Parameter Optimization for Graph Nav

## System Dependencies

1. [gflags](https://github.com/gflags/gflags)
1. [LibTorch](https://pytorch.org/get-started/locally/): Requires the cxx11 ABI version libtorch. Unpack the libtorch zip
   file to `/opt/libtorch`.
1. *Optional, if using libtorch with an Nvidia GPU*:   
    Install **both** [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) - these are separate installations.

# To Build

```
cmake .
make -j
```

# To Run

In order to train the learned trajectory selector, one has to generate data from a bag file. This can be done with the following commands:

'''
python3 bag_to_csv.py <bag_filename>.bag
python3 gen_dataset.py -d <bag_filename>  # make sure the constants in this file match the constants found in ut_automata/src/vesc_driver/vesc_driver.cpp when the bag was recorded.
./bin/param_data_gen -dataset_name="<bag_filename>/dataset.csv" -output="<bag_filename>/state_matrix.csv"
python3 network.py -d "<bag_filename>/state_matrix.csv"
```
