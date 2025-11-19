---
title: Ranx
subtitle: A Modern C++ Parallel Random Number Generator
description: Ranx (RANdom neXt) is a next-generation parallel algorithmic (pseudo) random number generator
---

[![GitHub Release](https://img.shields.io/github/v/release/arminms/ranx?logo=github&logoColor=lightgray)](https://github.com/arminms/ranx/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/ranx/HEAD?labpath=01_randomness_primer.ipynb)

---

::::{grid} 1 1 2 2

:::{grid-item}

Ranx is a next-generation parallel [algorithmic (pseudo) random number generator](wiki:Pseudorandom_number_generator) available as both a utility, as well as a modern <wiki:header-only> <wiki:C++> library supporting <wiki:OpenMP>, <wiki:CUDA>, <wiki:ROCm> and <wiki:oneAPI>.

As a library, Ranx provides alternatives to [STL](wiki:Standard_Template_Library)'s [`std::generate()`](https://en.cppreference.com/w/cpp/algorithm/generate) family of algorithms that exclusively designed for parallel random number generation on CPUs and GPUs.

:::

:::{grid-item}

```{image} ./images/ranx_logo.svg
:label: ranx-logo
```

:::

::::

:::::{aside}

::::{important} Try Ranx in a Container
:class: dropdown
[Docker:](wiki:Docker_(software))
```bash
docker run -p 8888:8888 -it --rm asobhani/ranx
```
[Apptainer:](wiki:Singularity_(software))
```bash
apptainer run docker://asobhani/ranx:latest
```
::::

:::::{seealso} Try Ranx on Binder
:class: dropdown
::::{grid} 2 2 2 2
:::{grid-item}
👉   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arminms/ranx/HEAD?labpath=01_randomness_primer.ipynb)
:::
:::{grid-item}
_Be advised sometimes it takes several minutes to start!_
:::
::::

:::::

---

## Features at a Glance 🔮

::::{grid} 1 1 2 2

:::{card}
:header: 🧮 [Support both CPU and GPU](./04_working_with_ranx.md#ranx_tranformation_tab)
:footer: [Learn more »](./04_working_with_ranx.md)

Ranx uses [block splitting](https://github.com/arminms/ranx/blob/main/include/ranx/algorithm/generate.hpp#L320-L330) and [leapfrog](https://github.com/arminms/ranx/blob/main/include/ranx/algorithm/generate.hpp#L31-L41) algorithms for parallel random number generation on CPU and GPU, respectively.
:::

:::{card}
:header: 🎲 [Play fair](./04_working_with_ranx.md#check_fair_play)
:footer: [Learn more »](./04_working_with_ranx.md)

Using the same seed, Ranx always generates the same sequence, independent of the number of parallel threads and the underlying hardware on all platforms.
:::

::::

---

(features)=
## Key Features 🥇
- 🖥️ Multiplatform (<wiki:Linux>, <wiki:macOS>, <wiki:Windows>)
- 🛠️ Support four target APIs (<wiki:OpenMP>, <wiki:CUDA>, <wiki:ROCm>, [oneAPI](wiki:OneAPI_(compute_acceleration)))
- 🎲 Play fair on all supported platforms
- 💥 No dependencies
- 🖇️ Header-only
- 🎱 Include [PCG family](./04_working_with_ranx.md#supported_engines) as engine
- 📊 Include 32 distributions provided by [TRNG](./04_working_with_ranx.md#supported_distributions) library
- 📥 Easily integrates with existing libraries and code (via <wiki:CMake> configs)
- 🧪 Include unit tests using [`Catch2`](https://github.com/catchorg/Catch2)
- 🏃 Include benchmarks using [`Google Benchmark`](https://github.com/google/benchmark)
- 📖 Well documented