## Requirements

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 11 GB VRAM (we used RTX 2080Ti)

### Software Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio for Windows, GCC for Linux)
- CUDA SDK for PyTorch extensions, install *after* Visual Studio or GCC
- C++ Compiler and CUDA SDK must be compatible
- FFMPEG to create result videos

### Additional python packages

- RoMa (for rotation representations)
- DearPyGUI (for viewer interface)
- NVDiffRast (for mesh rendering in viewer)

### Tested Platforms

| PyTorch Version | CUDA version | Linux | Windows (VS2022) | Windows (VS2019) |
|-| - | - | - | - |
| 2.0.1 | 11.7.1 | Pass | Fail to compile | Pass |
| 2.2.0 | 12.1.1 | Pass | Pass | Pass |

## Installation

Our default installation method is based on Conda package and environment management:

### 1. Create conda environment and install CUDA

```shell
git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
cd GaussianAvatars

conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja  # use the right CUDA version
```

### 2. Setup paths

#### For Linux

```shell
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation
```

#### For Windows with PowerShell

```shell
conda env config vars set CUDA_PATH="$env:CONDA_PREFIX"  

## Visual Studio 2022 (modify the version number `14.39.33519` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;$env:PATH"
## or Visual Studio 2019 (modify the version number `14.29.30133` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;$env:PATH" 

# re-activate the environment to make the above eonvironment variables effective
conda deactivate
conda activate gaussian-avatars
```

#### For Windows with Command Prompt

```shell
conda env config vars set CUDA_PATH=%CONDA_PREFIX%

## Visual Studio 2022 (modify the version number `14.39.33519` accordingly)
conda env config vars set PATH="%CONDA_PREFIX%\Script;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;%PATH%"
## or Visual Studio 2019 (modify the version number `14.29.30133` accordingly)
conda env config vars set PATH="%CONDA_PREFIX%\Script;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;%PATH%"

# re-activate the environment to make the above eonvironment variables effective
conda deactivate
conda activate gaussian-avatars
```

### 3. Install PyTorch and other packages

```shell
# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# or
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

# Install the rest packages (can take a while to compile diff-gaussian-rasterization, simple-knn, and nvdiffrast)
pip install -r requirements.txt
```
