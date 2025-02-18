[← Back to Main](../readme/readme.md#installation)

# Basic Installation

The following is necessary to create your conda environment for running PREFFECT.

## Setup of PREFFECT Conda Environment

We recommend Conda for managing the installation of the necessary libraries required for this project. To facilitate a straightforward setup, YAML files that contain all the requisite libraries are provided in `./ymls/`. We provide two separate YAML files. 
- **_preffect_CPU_LATEST.yml_** is expected to work on most Linux/Unix systems, and performs all tasks on the CPU. 
- **_preffect_GPU_LATEST.yml_** takes advantage of available GPUs for accelerated training and analysis. 

Before proceeding with the GPU YAML, ensure that your system is equipped with a compatible NVIDIA GPU and that the [appropriate NVIDIA drivers and CUDA toolkit are installed](https://developer.nvidia.com/cuda-downloads). You must also ensure that the installation of PyTorch is compatible with your GPU.

Execute the following command in your terminal:
```
conda env create --file preffect_{CPU,GPU}_LATEST.yml 
```

<br>
This will create an environment named `preffect_{CPU,GPU}_env`, which can be activated with the following command:

```
conda activate preffect_{CPU,GPU}_env
```

## Troubleshooting
There have been reports where the package `pyg` (torch geometric) can error when installing the YAML file. If this occurs, remove the package from the YAML file and re-run the installation command. Then, install the troublesome package separately within the same conda environment:

```
conda install pyg -c pyg
```


## Creating a Kernel (e.g. VSCODE or Jupyter)
```
python -m ipykernel install --user --name=preffect_{cpu,gpu}_env
```

##
[← Back to Main](../readme/readme.md#installation)