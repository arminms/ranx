//
// ranx - Display random numbers using the ranx library
//
// A C++ implementation inspired by Ubuntu's rand utility
// Copyright (c) 2025 Armin Sobhani (https://arminsobhani.ca)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
    #include <thrust/universal_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
#endif

#include <ranx/random>


const std::string VERSION = "1.0.0";
const std::string PROGRAM_NAME = "ranx";

template<typename T>
class no_init
{   static_assert(
    std::is_fundamental<T>::value,
    "should be a fundamental type");
public: 
    // constructor without initialization
    no_init () noexcept {}
    // implicit conversion T → no_init<T>
    RANX_DEVICE_CODE
    constexpr  no_init (T value) noexcept: v_{value} {}
    // implicit conversion no_init<T> → T
    RANX_DEVICE_CODE
    constexpr  operator T () const noexcept { return v_; }
    private:
    T v_;
};


void print_version()
{
    std::cout << PROGRAM_NAME << " version " << VERSION << "\n"
              << "A parallel random number generator using the ranx library\n"
              << "Copyright (c) 2025 Armin Sobhani\n"
              << "License: MIT\n";
}

void print_help()
{
    std::cout << "Usage: " << PROGRAM_NAME << " [OPTION]\n\n"
              << "Write random numbers to standard output.\n\n"
              << "Options:\n"
              << "  -N count           the count of random numbers (default=1)\n"
              << "  -L, --min number   the lower limit of the random numbers (default=0)\n"
              << "  -M, --max number   the upper limit of the random numbers (default=32576)\n"
              << "  -u, --unique       generate unique numbers (non duplicate values)\n"
              << "  -f                 generate float numbers from 0 to 1\n"
              << "  -p precision       the precision of float numbers (activates -f)\n"
              << "  -s number          the seed for the random numbers generator\n"
              << "                     (default: current time)\n"
              << "  -d STRING          delimiter between the numbers (default SPACE)\n"
              << "  --eof STRING       what to print at the end (default newline)\n"
              << "  --bof STRING       what to print at the beginning (default nothing)\n"
              << "  --help             display this help and exit\n"
              << "  --version          output version information and exit\n\n"
              << "Examples:\n"
              << "  " << PROGRAM_NAME << " -N 10              Generate 10 random numbers\n"
              << "  " << PROGRAM_NAME << " -N 5 -M 100        Generate 5 random numbers from 0 to 100\n"
              << "  " << PROGRAM_NAME << " -N 10 -u -M 20     Generate 10 unique numbers from 0 to 20\n"
              << "  " << PROGRAM_NAME << " -f -p 4 -N 5       Generate 5 float numbers with 4 decimals\n"
              << "  " << PROGRAM_NAME << " -N 5 -d \", \"       Generate 5 numbers separated by commas\n";
}

std::string process_escape_sequences(const std::string& str)
{
    std::string result;
    for (size_t i = 0; i < str.length(); ++i)
    {
        if (str[i] == '\\' && i + 1 < str.length())
        {
            switch (str[i + 1])
            {
                case 'n': result += '\n'; ++i; break;
                case 't': result += '\t'; ++i; break;
                case 'r': result += '\r'; ++i; break;
                case '\\': result += '\\'; ++i; break;
                default: result += str[i]; break;
            }
        }
        else
        {
            result += str[i];
        }
    }
    return result;
}

int main(int argc, char* argv[])
{
    // Default parameters
    size_t count = 1;
    int min_value = 0;
    int max_value = 32576;
    bool unique = false;
    bool generate_float = false;
    int precision = 6;
    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::string delimiter = " ";
    std::string eof_string = "\n";
    std::string bof_string = "";

    std::ios_base::sync_with_stdio(false);  // Disable sync with C stdio
    std::cin.tie(nullptr);                  // Untie cin from cout

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {   print_help();
            return 0;
        }
        else if (arg == "--version")
        {   print_version();
            return 0;
        }
        else if (arg == "-N" && i + 1 < argc)
            count = std::stoull(argv[++i]);
        else if ((arg == "-L" || arg == "--min") && i + 1 < argc)
            min_value = std::stoi(argv[++i]);
        else if ((arg == "-M" || arg == "--max") && i + 1 < argc)
            max_value = std::stoi(argv[++i]);
        else if (arg == "-u" || arg == "--unique")
            unique = true;
        else if (arg == "-f")
            generate_float = true;
        else if (arg == "-p" && i + 1 < argc)
        {   precision = std::stoi(argv[++i]);
            generate_float = true;
        }
        else if (arg == "-s" && i + 1 < argc)
            seed = std::stoull(argv[++i]);
        else if (arg == "-d" && i + 1 < argc)
            delimiter = process_escape_sequences(argv[++i]);
        else if (arg == "--eof" && i + 1 < argc)
            eof_string = process_escape_sequences(argv[++i]);
        else if (arg == "--bof" && i + 1 < argc)
            bof_string = process_escape_sequences(argv[++i]);
        else
        {   std::cerr << PROGRAM_NAME << ": invalid option '" << arg << "'\n"
                      << "Try '" << PROGRAM_NAME << " --help' for more information.\n";
            return 1;
        }
    }

    // Validate parameters
    if (unique && count > static_cast<size_t>(max_value + 1))
    {   std::cerr << PROGRAM_NAME << ": error: cannot generate " << count
                  << " unique numbers with max value " << max_value << "\n";
        return 1;
    }

    // Print beginning of file string
    if (!bof_string.empty())
        std::cout << bof_string;

    if (generate_float)
    {   // Generate floating point numbers
#if defined(__CUDACC__)
        // thrust::universal_vector<float> numbers(count);
        thrust::device_vector<no_init<float>> d_numbers(count);
        ranx::cuda::generate_n
        (   d_numbers.begin()
#elif defined(__HIP_PLATFORM_AMD__)
        thrust::device_vector<no_init<float>> d_numbers(count);
        ranx::rocm::generate_n
        (   d_numbers.begin()
#else
        std::vector<no_init<float>> numbers(count);
        ranx::generate_n
        (   std::begin(numbers)
#endif
        ,   count
        ,   ranx::bind(trng::uniform01_dist<float>(), pcg32(seed))
        );
#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
        thrust::host_vector<no_init<float>> numbers(count);
        thrust::copy(d_numbers.begin(), d_numbers.end(), numbers.begin());
#endif
        std::cout << std::fixed << std::setprecision(precision);
        for (size_t i = 0; i < count; ++i)
        {   if (i > 0) std::cout << delimiter;
            std::cout << numbers[i];
        }
    }
    else if (unique)
    {   // Generate unique integers
        if (count > static_cast<size_t>(max_value - min_value + 1))
        {   std::cerr << PROGRAM_NAME << ": error: cannot generate more unique numbers than max-min+1\n";
            return 1;
        }

        // Create a vector with all possible values
        std::vector<no_init<int>> all_numbers(max_value - min_value + 1);
        std::iota(all_numbers.begin(), all_numbers.end(), min_value);

        // Shuffle using standard algorithm
        std::shuffle
        (   std::begin(all_numbers)
        ,   std::end(all_numbers)
        ,   pcg32(seed)
        );

        // Take the first 'count' numbers
        for (size_t i = 0; i < count; ++i)
        {   if (i > 0) std::cout << delimiter;
            std::cout << all_numbers[i];
        }
    }
    else
    {   // Generate regular integers
#if defined(__CUDACC__)
        thrust::device_vector<no_init<int>> d_numbers(count);
        ranx::cuda::generate_n
        (   d_numbers.begin()
#elif defined(__HIP_PLATFORM_AMD__)
        thrust::device_vector<no_init<int>> d_numbers(count);
        ranx::rocm::generate_n
        (   d_numbers.begin()
#else
        std::vector<no_init<int>> numbers(count);
        ranx::generate_n
        (   std::begin(numbers)
#endif
        ,   count
        ,   ranx::bind(trng::uniform_int_dist(min_value, max_value), pcg32(seed))
        );
#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
        thrust::host_vector<no_init<int>> numbers(count);
        thrust::copy(d_numbers.begin(), d_numbers.end(), numbers.begin());
#endif
        for (size_t i = 0; i < count; ++i)
        {   if (i > 0) std::cout << delimiter;
            std::cout << numbers[i];
        }
    }

    // Print end of file string
    std::cout << eof_string;

    return 0;
}
