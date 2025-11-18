---
title: Ranx Command Line Tool
subtitle: A parallel random number generator utility
description: Guide to Use Ranx Command Line Tool
kernelspec:
  name: xcpp11
  display_name: C++11
---

# Ranx Command Line Tool

---

Ranx command line tool is a parallel random number generator utility built on the Ranx library. It provides functionality similar to the Ubuntu `rand` utility but leverages the high-performance, reproducible random number generation capabilities of the Ranx library.

## Features

- **Fast parallel random number generation** using the PCG family of generators
- **Reproducible sequences** across all platforms when using the same seed
- **Multiple output formats**: integers, floats, unique values
- **Flexible formatting**: custom delimiters and output boundaries
- **High-quality randomness** using [PCG32](https://www.pcg-random.org/) engine with [TRNG](https://github.com/rabauke/trng4) distributions

## Building

The utility is built automatically when building the Ranx library (unless disabled):

```bash
cmake -S . -B build
cmake --build build -j
```

To disable building the utility:

```bash
cmake -S . -B build -DRANX_BUILD_UTILITY=OFF
```

## Installation

```bash
cmake --install build
```

This installs the `ranx` executable to your system's binary directory (typically `/usr/local/bin`).

## Usage

```
ranx [OPTION]
```

### Options

- `-N count` - Number of random numbers to generate (default=1)
- `-L, --min number` - The lower limit of the random numbers (default=0)
- `-M, --max number` - The upper limit of the random numbers (default: 32576)
- `-u, --unique` - Generate unique numbers without duplicates
- `-f` - Generate floating-point numbers between 0 and 1
- `-p precision` - Set decimal precision for floats (activates `-f`)
- `-s number` - Set the random seed (default: current time)
- `-d STRING` - Delimiter between numbers (default: space)
- `--bof STRING` - String to print at the beginning
- `--eof STRING` - String to print at the end (default: newline)
- `--help` - Display help message
- `--version` - Show version information

### Examples

Generate 10 random numbers:
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 10
```

Generate 5 numbers from 0 to 100 (closed range):
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 5 -M 100
```

Generate 10 unique numbers from 10 to 20 (closed range):
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 10 -u -L 10 -M 20
```

Generate 5 floating-point numbers with 4 decimal places:
```{code-cell} cpp
:tags: [hide-output]
!ranx -f -p 4 -N 5
```

Generate numbers separated by commas:
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 5 -d ", "
```

Generate reproducible sequence with a specific seed:
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 10 -s 42
```

Format as a JSON array:
```{code-cell} cpp
:tags: [hide-output]
!ranx -N 5 -d ", " --bof "[" --eof "]"
```

## Technical Details

### Random Number Engine

The utility uses the **PCG32** (Permuted Congruential Generator) engine from the PCG family, which provides:
- Excellent statistical properties
- Fast generation speed
- Small state size
- Reproducible sequences

### Distributions

- **Integers**: Uses `trng::uniform_int_dist` for uniform integer distribution
- **Floats**: Uses `trng::uniform01_dist` for uniform distribution between 0 and 1
- **Unique values**: Uses Fisher-Yates shuffle via `std::shuffle` with PCG32 (not parallel yet)

### Reproducibility

When you provide the same seed with `-s`, the utility guarantees identical output on all supported platforms (Linux/macOS/Windows):

```{code-cell} cpp
:tags: [hide-output]
!ranx -N 5 -s 123
```

## Comparison with Ubuntu `rand`

This implementation provides similar functionality to the Ubuntu `rand` utility with some enhancements:

### Similarities
- Command-line interface and option names
- Support for integer and float generation
- Custom delimiters and formatting options
- Seed-based reproducibility

### Differences
- Does support a new flag for the low limit (`-L`/`--min` flags)
- Uses high-quality PCG32 engine (vs. standard C library RNG)
- Built on the ranx parallel generation library
- Does not support backslash escape interpretation (`-e`/`-E` flags)
- Does not support mask formatting (`--mask` flag)

## License

MIT License - Copyright (c) 2025 Armin Sobhani

## See Also

- [PCG Random Number Generators](https://www.pcg-random.org/)
- [TRNG library](https://github.com/rabauke/trng4)
- [Ubuntu rand utility](https://manpages.ubuntu.com/manpages/xenial/man1/rand.1.html)
