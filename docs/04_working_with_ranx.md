---
title: Working with Ranx
subtitle: Learning Ranx API
subject: Ranx Quickstart Tutorial
description: All you have to know about using Ranx library.
kernelspec:
  name: xcpp17-openmp
  display_name: C++17-OpenMP
---

# Working with Ranx

---

Ranx is a modern header-only *C++* library for parallel algorithmic random number generation. Using [*block splitting*](#block_splitting) on CPUs (<wiki:OpenMP>) and [*leapfrogging*](#leapfrog) on GPUs (<wiki:CUDA>/<wiki:ROCm>/[oneAPI](wiki:OneAPI_(compute_acceleration))), paired with distributions from [*TRNG library*](https://github.com/rabauke/trng4) that avoid discarding values, Ranx wraps the *engine*+*distribution* into a device‑compatible functor and applies jump‑ahead/stride patterns provided by [`PCG generators`](wiki:Permuted_congruential_generator) so that, given the same seed, you get reproducible sequences independent of thread count or backend. In other words, Ranx fulfills all the necessary and sufficient conditions to [*play fair*](./01_randomness_primer.md#fairplay) on all supported platforms.

(ranx_101)=
## Ranx 101

To start using Ranx in your code, you just need to include the header:

```cpp
#include <ranx/random>
```

That will also include all the engines and the distributions that come with it. Check [here](./quickstart.md#developing_w_ranx) if you're compiling for different backends.

(supported_engines)=
### Supported engines

Currently, Ranx includes all the generators from [`PCG family`](wiki:Permuted_congruential_generator) (variation of [LCG](wiki:Linear_congruential_generator)), mainly because their `discard(n)` function takes $O(log\ n)$ to complete:

::::{grid} 1 1 2 2

:::{card}
:header: 1️⃣ [32-Bit Generators with 64-Bit State](https://www.pcg-random.org/using-pcg-cpp.html#bit-generators-with-64-bit-state)
- `pcg32`
- `pcg32_oneseq`
- `pcg32_unique`
- `pcg32_fast`
:::

:::{card}
:header: 2️⃣ [64-Bit Generators with 128-Bit State](https://www.pcg-random.org/using-pcg-cpp.html#bit-generators-with-128-bit-state)
- `pcg64`
- `pcg64_oneseq`
- `pcg64_unique`
- `pcg64_fast`
:::

::::

You can also use [STL](wiki:Standard_Template_Library)'s engines with Ranx, if they provide `discard(n)` member function. But they may neither perform well in parallel (their `discard(n)` is mostly $O(n)$, if any)  nor play fair as the implementation can be platform-dependent (e.g. [g++](wiki:GNU_Compiler_Collection) vs. [Visual C++](wiki:Microsoft_Visual_C++))

Support for [`std::philox_engine`](https://en.cppreference.com/w/cpp/numeric/random/philox_engine.html) will be added to Ranx in the near future.

(supported_distributions)=
### Supported distributions

Ranx includes all the *32 distributions* provided by [TRNG](https://github.com/rabauke/trng4) library. You can also use [STL](wiki:Standard_Template_Library)'s distribution with Ranx, but again, they don't warrant *fair play* as they may discard some values or have platform-dependent implementations.

::::{grid} 1 1 2 2

:::{card}
:header: 1️⃣ Bernoulli distributions
- `trng::bernoulli_dist`
- `trng::binomial_dist`
- `trng::negative_binomial_dist`
- `trng::geometric_dist`
- `trng::hypergeometric_dist`
:::

:::{card}
:header: 2️⃣ Normal distributions
- `trng::normal_dist`
- `trng::lognormal_dist`
- `trng::cauchy_dist`
- `trng::chi_square_dist`
- `trng::correlated_normal_dist`
- `trng::logistic_dist`
- `trng::maxwell_dist`
- `trng::rayleigh_dist`
- `trng::truncated_normal_dist`
- `trng::student_t_dist`
:::

:::{card}
:header: 3️⃣ Uniform distributions
- `trng::uniform01_dist`
- `trng::uniform_dist`
- `trng::uniform_int_dist`
:::

:::{card}
:header: 4️⃣ Sampling distributions
`trng::discrete_dist`
`trng::fast_discrete_dist`
:::

:::{card}
:header: 5️⃣ Poisson distributions
- `trng::poisson_dist`
- `trng::exponential_dist`
- `trng::gamma_dist`
- `trng::weibull_dist`
- `trng::extreme_value_dist`
- `trng::zero_truncated_poisson_dist`
:::

:::{card}
:header: 6️⃣ Miscellaneous distributions
- `trng::beta_dist`
- `trng::pareto_dist`
- `trng::powerlaw_dist`
- `trng::snedecor_f_dist`
- `trng::tent_dist`
- `trng::twosided_exponential_dist`
:::

::::

### Changing existing code to use Ranx

Let’s start with the [serial code](./02_serial_random_number_generation.md#generate_n_w_bind) we introduced in the previous chapters and transform it to use Ranx for different parallel APIs/ecosystems. Let's first include the necessary headers and function templates as we usually do:

```{code-cell} cpp
// setting OpenMP headers and library required by Ranx
#pragma cling add_include_path("/usr/lib/llvm-9/include/openmp")
#pragma cling load("libomp.so.5")
```
+++
```{code-cell} cpp
#include <iostream>    // <-- std::cout and std::endl
#include <iomanip>     // <-- std::setw()
#include <g3p/gnuplot> // <-- g3p::gnuplot

// function template to print the numbers
template <typename RandomIterator>
void print_numbers(RandomIterator first, RandomIterator last)
{   auto n = std::distance(first, last);
    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
        std::cout << '\n';
        std::cout << std::setw(3) << *(first + i);
    }
    std::cout << '\n' << std::endl;
}

// function template to render two randograms side-by-side
template<typename Gnuplot, typename RandomIterator>
void randogram2
(   const Gnuplot& gp
,   RandomIterator first
,   RandomIterator second
,   size_t width = 200
,   size_t height = 200
)
{   gp  ("set term pngcairo size %d,%d", width * 2, height)
        ("set multiplot layout 1,2")
        ("unset key; unset colorbox; unset tics")
        ("set border lc '#333333'")
        ("set margins 0,0,0,0")
        ("set bmargin 0; set lmargin 0; set rmargin 0; set tmargin 0")
        ("set origin 0,0")
        ("set size 0.5,1")
        ("set xrange [0:%d]", width)
        ("set yrange [0:%d]", height)
        ("plot '-' u 1:2:3:4:5 w rgbimage");
    for (size_t i = 0; i < width; ++i)
        for (size_t j = 0; j < height; ++j)
        {   int c = *first++;
            gp << i << j << c << c << c << "\n";
        }
    gp.end() << "plot '-' u 1:2:3:4:5 w rgbimage\n";
    for (size_t i = 0; i < width; ++i)
        for (size_t j = 0; j < height; ++j)
        {   int c = *second++;
            gp << i << j << c << c << c << "\n";
        }
    gp.end() << "unset multiplot\n";
    display(gp, false);
}
```
::::{tab-set}
:label: ranx_tranformation_tab

:::{tab-item} Serial

```{code-cell} cpp
#include <vector>     // <-- std::vector
#include <random>     // <-- std::t19937 and std::uniform_int_distribution
#include <algorithm>  // <-- std::generate() and std::generate_n()
#include <functional> // <-- std::bind() and std::ref()

const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
std::mt19937 r(seed);
std::uniform_int_distribution<int> u(10, 99);

std::generate_n
(   std::begin(v)
,   n
,   std::bind(u, std::ref(r))
);


print_numbers(std::begin(v), std::end(v));
```
:::

:::{tab-item} OpenMP
```{code-cell} cpp
:label: ranx_openmp_code
#include <vector>      // <-- std::vector
#include <ranx/random> // <-- ranx::generate_n(), ranx::bind(), pcg32, trng



const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
pcg32 r(seed);
trng::uniform_int_dist u(10, 99);

ranx::generate_n
(   std::begin(v)
,   n
,   ranx::bind(u, r)
);


print_numbers(std::begin(v), std::end(v));
```
:::

:::{tab-item} CUDA/ROCm
```{code-cell} cpp
:tags: [skip-execution]
#include <vector>      // <-- std::vector
#include <ranx/random> // <-- ranx::generate_n(), ranx::bind(), pcg32, trng
#include <thrust/device_vector.h>


const unsigned long seed{2718281828};
const auto n{100};
thrust::device_vector<int> v(n);
pcg32 r(seed);
trng::uniform_int_dist u(10, 99);

ranx::cuda::generate_n
(   std::begin(v)
,   n
,   ranx::bind(u, r)
);


print_numbers(std::begin(v), std::end(v));
```
```{embed} #ranx_openmp_code
:remove-input: true
:remove-output: false
```
:::

:::{tab-item} oneAPI
```{code-cell} cpp
:tags: [skip-execution]

// no need for std::vector
#include <ranx/random> // <-- ranx::generate_n(), ranx::bind(), pcg32, trng
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>

const unsigned long seed{2718281828};
const auto n{100};
sycl::buffer<int> v(sycl::range(n));
pcg32 r(seed);
trng::uniform_int_dist u(10, 99);

ranx::oneapi::generate_n
(   std::begin(v)
,   n
,   ranx::bind(u, r)
);

sycl::host_accessor va{v, sycl::read_only};
print_numbers(std::begin(va), std::end(va));
```
```{embed} #ranx_openmp_code
:remove-input: true
:remove-output: false
```
:::

::::

To cut a long story short, for the part related to Ranx, you just need to change `std::generate()`/`std::generate_n()` and `std::bind()` to the corresponding Ranx alternatives `ranx::generate()`/`ranx::generate_n()` and `ranx::bind()`.

(check_fair_play)=
## Checking if it plays fair

Now lets see if we can get the same randogram as the serial code, using the same *seed*/*engine*/*distribution* triplet for the parallel version:

```{code-cell} cpp
:label: ranx_randograms

const size_t w{240}, h{240}, n{w * h};
std::vector<int> parallel(n), serial(n);
pcg32 pr(seed), sr(seed);  // start with the same engine and seed
trng::uniform_int_dist c(0, 255); // for rgb

// parallel version passing copy of the engine
ranx::generate_n(std::begin(parallel), n, ranx::bind(c, pr));
std::generate_n(std::begin(serial), n, std::bind(c, std::ref(sr)));

// instantiate the gnuplot
g3p::gnuplot gp;

// rendering two randograms side-by-side for comparison
randogram2(gp, std::begin(parallel), std::begin(serial), w, h);
```

## Benchmarks

Here's come the moment of truth. Let's see if our parallel versions can actually outperform the serial version. Let's first initialize our containers:

```{code-cell} cpp
#include <thread>
#include <execution>

#pragma cling load("libtbb.so.2")

const size_t n = 1'000'000;
std::hash<std::thread::id> hasher;
std::vector<int> s(n), rs(n), p(n), bs(n);
pcg32 r{seed}; // use the reference for the serial version
```
::::{tab-set}
:label: benchmark_tab

:::{tab-item} Serial

```{code-cell} cpp
%%timeit
std::generate_n
(
    std::begin(s)
,   n
,   std::bind(u, std::ref(r))
)


;
```

:::

:::{tab-item} Random seeding

```{code-cell} cpp
%%timeit
std::generate_n
(   std::execution::par
,   std::begin(rs)
,   n
,   [&]()
{   thread_local pcg32 r(hasher(std::this_thread::get_id()));
    return u(r);
}
);
```

:::

:::{tab-item} Parametrization

```{code-cell} cpp
%%timeit
std::generate_n
(   std::execution::par
,   std::begin(p)
,   n
,   [&]()
{   thread_local pcg32 r(seed, hasher(std::this_thread::get_id()));
    return u(r);
}
);
```

:::

:::{tab-item} Block splitting (Ranx)

```{code-cell} cpp
%%timeit
ranx::generate_n
(
    std::begin(bs)
,   n
,   ranx::bind(u, pcg32{seed})
)


;
```

:::

::::
