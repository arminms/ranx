---
title: Start using Ranx
subject: Ranx Quickstart
subtitle: CMake integration and rapid prototyping with Jupyter
short_title: Get Started
description: Ranx is available through GitHub.
kernelspec:
  name: xcpp17
  display_name: C++17
---

# Start using Ranx

---

## Requirements

- <wiki:C++> compiler supporting the `C++17` standard (e.g. [GCC](wiki:GNU_Compiler_Collection) 9.3 or higher)
- <wiki:CMake> version 3.21 or higher.

And the following optional third-party libraries:
* [Catch2](https://github.com/catchorg/Catch2) v3.1 or higher for unit testing
* [Google Benchmark](https://github.com/google/benchmark) for benchmarks

The <wiki:CMake> script configured in a way that if it cannot find the optional third-party libraries it tries to fetch and build them automatically. So, there is no need to do anything if they are missing but you need an internet connection for that to work.

## Building from source

Once you have all the requirements you can build and install it using the
following commands:

```{code} bash
:label: cmake-build
git clone https://github.com/arminms/ranx.git
cd ranx
cmake -S . -B build && cmake --build build
cmake --install build
```

::::{attention} Installing into non system folders 🖥️

The last command on <wiki:Linux> or <wiki:macOS> must be preceded by `sudo`, and on <wiki:Windows> must be run as an administrator unless you add `--prefix` option at the end to change the default installation path to a none system folder (e.g. `cmake --install build --prefix ~/.local`).
::::

::::{hint} Running unit tests 🧪
:class: dropdown

Use the following command after building if you like to run the unit tests as well:

```bash
cd build && ctest
```
::::

::::{hint} Running benchmarks 🏃
:class: dropdown

Use the following command after building if you like to run the benchmarks:

```bash
cd build && perf/benchmarks --benchmark_counters_tabular=true
```
::::

## Developing with Ranx

Ranx exports four (namespaced) <wiki:CMake> targets and also <wiki:CMake> config scripts for downstream applications:

- `ranx::cuda`
- `ranx::oneapi`
- `ranx::openmp`
- `ranx::rocm`

Linking against them adds the proper include paths and links your target with
proper libraries depending on the API. This means if your project also relies on <wiki:CMake> and Ranx has been installed on your system, a better option is to use [`find_package()`](xref:cmake#command/find_package) in your project's `CMakeLists.txt` as shown below:

```cmake
find_package(ranx CONFIG COMPONENTS openmp cuda)

# link test.cpp with ranx using OpenMP API
add_executable(test_openmp test.cpp)
target_link_libraries(test_openmp PRIVATE ranx::openmp)

# link test.cu with ranx using CUDA API
add_executable(test_cuda test.cu)
target_link_libraries(test_cuda PRIVATE ranx::cuda)
```

::::{note} Embedding with CMake 📥

To embed the library directly into an existing <wiki:CMake> project, you can mix [`find_package()`](xref:cmake#command/find_package) with [`FetchContent`](xref:cmake#module/FetchContent) module available on <wiki:CMake> `3.14` and higher:

```cmake
# include the module
include(FetchContent)

# first check if ranx is already installed
find_package(ranx CONFIG COMPONENTS oneapi)

# if not, try to fetch and make it available
if(NOT ranx_FOUND)
  message(STATUS "Fetching ranx library...")
  FetchContent_Declare(
    ranx
    GIT_REPOSITORY https://github.com/arminms/ranx.git
    GIT_TAG main
  )
  # setting required ranx components
  set(RANX_COMPONENTS oneapi CACHE STRING "Required components")
  FetchContent_MakeAvailable(ranx)
endif()

# link test.cpp with ranx using oneapi as API
add_executable(test_oneapi test.cpp)
target_link_libraries(test_oneapi PRIVATE ranx::oneapi)
```

The above approach first tries to find an installed version of Ranx and if it cannot then tries to fetch it from the repository. You can find a complete example of the above approach in the [`example`](https://github.com/arminms/ranx/blob/main/example/CMakeLists.txt#L5-L28) folder.
::::

(jupyter-rapid-prototyping)=
## Rapid prototyping with *Jupyter*

You can experiment with Ranx in [Jupyter](wiki:Project_Jupyter) notebooks with [Xeus-Cling](xref:xeus-cling) kernel. That's a very convenient way for rapid prototyping.

Depending on your preference, you can use one/all of the following methods to work with Ranx in Jupyter.

### Creating a Conda/Mamba environment

The easiest way to install [Xeus-Cling](xref:xeus-cling) is to create an environment named `cling` using [Mamba](xref:mamba#installation/mamba-installation):

```bash
mamba create -n cling
mamba activate cling
```

Then you can install [Xeus-Cling](xref:xeus-cling) in this environment and its dependencies:

```bash
mamba install xeus-cling -c conda-forge
```

Next, you can use `mamba env list` command to find where the `cling` environment is installed and use the following commands to install Ranx in the `cling` environment:

```{code} bash
:label: ranx-zip
wget https://github.com/arminms/ranx/releases/latest/download/install.zip
unzip install.zip -d <PATH/TO/CLING/ENV>
```

(python_ve)=
### Creating a Python environment

If you're more adventurous, you can build [Xeus-Cling](xref:xeus-cling) along with all the dependencies from the source in a Python virtual environment using the bash script available [here](xref:jupyter-xc#build_virtual_env). As a bonus, you will get a newer version of <xref:cling> (`v1.1`) based on `llvm 16.0` that supports <wiki:C++20>, <wiki:OpenMP>, <wiki:CUDA>.

Once done, you can install Ranx in your Python environment by unpacking the [zip file](#ranx-zip) or using [CMake](#cmake-build):

```bash
git clone https://github.com/arminms/ranx.git
cd ranx
cmake -S . -B build && cmake --build build
cmake --install build --prefix /path/to/python/virtual/environment
```

The same approach was used to build the container images below.

(containers)=
### Using pre-built container images

#### [Docker](wiki:Docker_(software))
```bash
docker run -p 8888:8888 -it --rm asobhani/ranx
```
If you like to work with your notebooks outside the container (e.g. current folder), you can use the following command instead:
```bash
docker run -v $PWD:/home/jovyan -p 8888:8888 -it --rm asobhani/ranx
```
You can also use the above container image as the starting point for your custom-made docker image (e.g. `FROM asobhani/ranx:latest`).

#### [Apptainer](wiki:Apptainer)

If you're working on an HPC cluster, you can use [Apptainer](wiki:Singularity_(software)) instead:

```bash
apptainer run docker://asobhani/ranx:latest
```
