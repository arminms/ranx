---
title: Serial Random Number Generation
subtitle: How to generate random numbers in C++
subject: Ranx Quickstart Tutorial
description: How to generate random numbers in C++ using the traditional C functions and the modern engine+distribution model
kernelspec:
  name: xcpp11
  display_name: C++11
---

# Serial Random Number Generation

---

::::{important} TL;DR ✨
This chapter shows how to generate random numbers in standard *C++* before and after *C++11*, moving from `rand()`/`srand()` and the pitfalls of modulo-based range mapping to the modern *engine+distribution* model.

You will learn the essentials of `std::mt19937` with `std::uniform_int_distribution`, and progressively refactor examples from basic loops to `std::generate()`/`std::generate_n()` using lambdas and `std::bind(dist, std::ref(engine))`—the exact patterns we’ll reuse later for parallel generation with Ranx.

If you already master *C++11* RNGs and idioms, you can jump ahead to the \
[next chapter »](./03_parallel_random_number_generation.ipynb).
::::
+++
## Before *C++11*
Before *C++11*, there were only two functions for random number generation:

- [`srand()`](https://en.cppreference.com/w/c/numeric/random/srand) to seed the random number generator
- [`rand()`](https://en.cppreference.com/w/c/numeric/random/rand) commonly based on an [LCG](wiki:Linear_congruential_generator) to generate the next random number

```{code-cell} cpp
:tags: [hide-output]

#include <iostream> // <-- std::out and std::endl
#include <cstdlib>  // <-- srand() and rand()
#include <ctime>    // <-- time()

srand(time(NULL));
for (auto i = 0; i < 10; ++i)
  std::cout << rand() << std::endl;
```

:::{seealso} Exercise 🛠️

Every time you run the above cell you will get a different sequence of numbers. Try it now.
:::

And the common practice was to use *modulo operator* (`%`), like [the way we did](./01_randomness_primer.ipynb#randogram) in the previous chapter, to bring them into a range:

```{code-cell} cpp
:tags: [hide-output]

// generate 10 random numbers from 0 to 99 (a range of 100 values)
for (auto i = 0; i < 10; ++i)
  std::cout << rand() % 100 << ' ';
std::cout << std::endl;
```

Or an arbitrary range (with an offset):

```{code-cell} cpp
:tags: [hide-output]

// generate 10 random numbers from 5 to 14
size_t min = 5, max = 14;
for (auto i = 0; i < 10; ++i)
  // i.e. (rand() % 10) + 5;
  std::cout << (rand() % (max - min + 1)) + min << ' ';
std::cout << std::endl;
```

:::{danger} Modulo bias ⚠️
:class: dropdown
:open: true

Using the *modulo operator* to bring random numbers into a specific range is generally not a good idea due to the potential for modulo bias.

**Modulo bias explained:**

- *Uneven Distribution* – When the maximum value of the random number generator (e.g., `RAND_MAX` in *C/C++*) is not an exact multiple of the desired range size, applying the modulus operator will result in an uneven distribution of numbers within that range. Some numbers will appear more frequently than others.
**Example:** If `RAND_MAX` is 32767 and you want numbers in the range 0-99, `32767 % 100 = 67`. This means numbers from 0 to 67 will have an extra chance to be generated compared to numbers from 68 to 99, as they can be generated from multiple inputs to the modulus operator (e.g., 0, 100, 200... all result in 0 when modulo 100).
- *Low-Order Bit Issues* – Some [ARNGs](wiki:Pseudorandom_number_generator), particularly older or simpler ones like [LCGs](wiki:Linear_congruential_generator), exhibit less randomness in their lower-order bits. Using the modulus operator effectively isolates and utilizes these potentially less random lower bits, further compromising the quality of the "random" numbers.

**Consequences of modulo bias:**
- *Non-Uniformity* – The resulting numbers will not be uniformly distributed across the desired range, meaning some values are more likely to occur than others.
- *Reduced Quality of Randomness* – For applications requiring high-quality randomness (e.g., simulations, cryptography), modulo bias can introduce exploitable patterns or inaccuracies.

You can get rid of the modulo bias by using an engine with a distribution, introduced in *C++11*. We'll cover that in the next section.
:::
+++
## Since *C++11*
*C++11* brings with it a more complex random-number library that provides multiple *engines* and many well-known *distributions* adopted from [Boost](wiki:Boost_(C++_libraries)).**Random** library. *Engines* act as a source of randomness to create random *unsigned values*, which are uniformly distributed between a predefined *minimum* and *maximum*; and *distributions*, transform those values into random numbers:

::::{grid} 2 4 6 8

:::{card}
:header: 🎲 [Engine](https://en.cppreference.com/w/cpp/numeric/random.html#Predefined_random_number_generators)
:footer: [Learn more »](https://en.cppreference.com/w/cpp/numeric/random.html#Predefined_random_number_generators)

Source of randomness that create random *unsigned values*, which are uniformly distributed between a predefined *minimum* and *maximum*
:::

:::{card}
:header: 📊 [Distribution](https://en.cppreference.com/w/cpp/numeric/random.html#Random_number_distributions)
:footer: [Learn more »](https://en.cppreference.com/w/cpp/numeric/random.html#Random_number_distributions)

Transform values generated by the engine into random numbers according to a defined statistical [probability density function](wiki:Probability_density_function).
:::

::::
+++
### Simple demo
Let’s see how we can come up with a simple program to use an engine and a distribution to generate random numbers. As there are many ways to bake a cake, we'll do it in four different ways:

::::{grid} 4 6 8 8
:numbered:

:::{card}
:header: 1️⃣ [`for` loop](#for_loop)
:footer: [Compare codes »](#serial_rng_tab)

Notice the exitance of `100` in the output. The distributions in `C++11` random library accept closed ranges or intervals. It means unlike the modulo method, `10` and `100` are both included.
:::

:::{card}
:header: 2️⃣ [`std::generate()` + *lambda*](#generate_w_lambda)
:footer: [Compare codes »](#serial_rng_tab)

Notice the use of `&` in the lambda closure to pass the references.
That’s especially important for passing engines with huge [footprint](./01_preliminary_concepts.ipynb#arng_attributes) like <wiki:Mersenne_Twister>, but as not much for the distributions.
:::

:::{card}
:header: 3️⃣ [`std::generate()` + `std::bind()`](#generate_w_bind)
:footer: [Compare codes »](#serial_rng_tab)

This is a cleaner and more suitable way of passing both the (reference to) engine and the distribution to the [`generate()`](https://en.cppreference.com/w/cpp/algorithm/generate.html) function. Both [`std::bind()`](https://en.cppreference.com/w/cpp/utility/functional/bind.html) and [`std::ref()`](https://en.cppreference.com/w/cpp/utility/functional/ref.html) are defined in the [functional header](https://en.cppreference.com/w/cpp/header/functional.html).
:::

:::{card}
:header: 4️⃣ [`std::generate_n()` + `std::bind()`](#generate_n_w_bind)
:footer: [Compare codes »](#serial_rng_tab)

We can also use the [`std::generate_n()`](https://en.cppreference.com/w/cpp/algorithm/generate_n.html) algorithm equally well as sometimes it's more convenient to pass the number.
:::

::::

Let's first include the necessary headers and define our output function template:

```{code-cell} cpp
#include <iostream>   // <-- std::cout and std::endl
#include <iomanip>    // <-- std::setw()
#include <vector>     // <-- std::vector
#include <random>     // <-- std::t19937 and std::uniform_int_distribution
#include <algorithm>  // <-- std::generate() and std::generate_n()
#include <functional> // <-- std::bind() and std::ref()

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
```

And here are the actual codes to compare:

+++
**`for` loop**

```{code-cell} cpp
:label: for_loop

const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
std::mt19937 r(seed);
std::uniform_int_distribution<int> u(10, 100);

for (auto& a : v) // <-- range-based for loop (C++11)
    a = u(r);

// for (size_t i = 0; i < std::size(v); ++i) // <-- old way
//    v[i] = u(r);

print_numbers(std::begin(v), std::end(v));
```

**`generate()`+lambda**

```{code-cell} cpp
:label: generate_w_lambda

const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
std::mt19937 r(seed);
std::uniform_int_distribution<int> u(10, 100);

std::generate
(   std::begin(v)
,   std::end(v)
,   [&]() { return u(r); }
);

print_numbers(std::begin(v), std::end(v));
```

**`generate()`+`bind()`**

```{code-cell} cpp
:label: generate_w_bind

const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
std::mt19937 r(seed);
std::uniform_int_distribution<int> u(10, 100);

std::generate
(   std::begin(v)
,   std::end(v)
,   std::bind(u, std::ref(r))
);

print_numbers(std::begin(v), std::end(v));
```

**`generate_n()`+`bind()`**

```{code-cell} cpp
:label: generate_n_w_bind

const unsigned long seed{2718281828};
const auto n{100};
std::vector<int> v(n);
std::mt19937 r(seed);
std::uniform_int_distribution<int> u(10, 100);

std::generate_n
(   std::begin(v)
,   n
,   std::bind(u, std::ref(r))
);

print_numbers(std::begin(v), std::end(v));
```

## Concluding remarks
We came up with four different ways to generate random numbers using an engine and a distribution. The ones more important for us are the last two: [`std::generate()`](https://en.cppreference.com/w/cpp/algorithm/generate.html) and [`std::generate_n()`](https://en.cppreference.com/w/cpp/algorithm/generate_n.html) with [`std::bind()`](https://en.cppreference.com/w/cpp/utility/functional/bind.html). Why? Because our parallel random number generator library relies on that construct. But that's the story for the next chapter.

::::{grid} 2 2 2 2

:::{card}
:link: ./01_randomness_primer.ipynb
<div style="text-align: left">⬅️ Previous</div>
<div style="text-align: left">A Primer On Randomness</div>
:::

:::{card}
:link: ./03_parallel_random_number_generation.ipynb
<div style="text-align: right">Next ➡️</div>
<div style="text-align: right">Parallel Random Number Generation</div>
:::

::::

