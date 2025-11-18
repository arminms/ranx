//
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
#ifndef _RANX_UTILITY_HPP_
#define _RANX_UTILITY_HPP_

namespace ranx {

//=== randogram() ==============================================================

template<typename Gnuplot, typename RandomIterator>
void randogram
(   const Gnuplot& gp
,   RandomIterator first
,   size_t width = 200
,   size_t height = 200
)
{   gp  ("set term pngcairo size %d,%d", width, height)
        ("unset key; unset colorbox; unset border; unset tics")
        ("set margins 0,0,0,0")
        ("set bmargin 0; set lmargin 0; set rmargin 0; set tmargin 0")
        ("set origin 0,0")
        ("set size 1,1")
        ("set xrange [0:%d]", width)
        ("set yrange [0:%d]", height)
        ("plot '-' u 1:2:3:4:5 w rgbimage");
    for (size_t i = 0; i < width; ++i)
        for (size_t j = 0; j < height; ++j)
        {   int c = *first++;
            gp << i << j << c << c << c << "\n";
        }
    gp.end();
    display(gp, false);
}

//=== randogram2() =============================================================

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

//=== whitenoise() ==============================================================

template<typename Gnuplot, typename RandomIterator>
void whitenoise
(   const Gnuplot& gp
,   RandomIterator first
,   size_t frames = 10
,   size_t width = 200
,   size_t height = 200
,   bool webp = false
)
{   gp  ("set term %s enhanced animate size %d,%d", webp ? "webp" : "gif", width, height)
        ("unset key; unset colorbox; unset border; unset tics")
        ("set margins 0,0,0,0")
        ("set bmargin 0; set lmargin 0; set rmargin 0; set tmargin 0")
        ("set origin 0,0")
        ("set size 1,1")
        ("set xrange [0:%d]", width)
        ("set yrange [0:%d]", height)
    for (size_t frame = 0; frame < frames; ++frame)
    {   gp  ("plot '-' u 1:2:3:4:5 w rgbimage");
        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < height; ++j)
            {   int c = *first++;
                gp << i << j << c << c << c << "\n";
            }
        gp.end();
    }
    display(gp, false);
}

//=== histogram ================================================================

// template<typename Generator> struct histogram
// {   histogram(Generator g, size_t count = 100'000, float binwidth = 2.0f, float binstart = 0.0f)
//     :   _binwidth(binwidth), _binstart(binstart), _data(count)
//     {   // std::generate_n(std::begin(_data), count, g);
//         ranx::generate_n(std::begin(_data), count, g);
//         // ranx::generate_n(std::begin(_data), count, g);
//         // for (size_t i = 0; i < count; ++i)
//         //     _data.push_back(g());
//     }
//     template<typename Gnuplot>
//     void plot(const Gnuplot& gp) const
//     {   gp  ("binwidth = %f", _binwidth)
//             ("binstart = %f", _binstart)
//             ("bin(x) = binwidth * floor((x - binstart) / binwidth) + binstart + binwidth/2.0")
//             ("unset key; unset colorbox;")
//             ("set title tc '#555555'" )
//             ("set style line 101 lt 1 lc '#555555' dt '. '")
//             ("set grid ls 101")
//             ("set title 'Distribution Histogram'")
//             ("set ylabel 'Frequency' tc '#555555'")
//             ("set style fill solid 0.5")
//             ("set boxwidth binwidth * 0.9")
//             ("set xrange [binstart:binstart + 100]")
//             ("set yrange [0:*]")
//             ("set border lc '#555555'")
//             ("plot '-' u (bin($1)):(1) smooth frequency with boxes");
//         for (int v : _data)
//             gp << v << "\n";
//         // (*this)(gp);
//         gp.end();
//     }
//     template<typename Gnuplot>
//     void operator()(const Gnuplot& gp) const
//     {   for (int v : _data)
//             gp << v << "\n";
//     }

// private:
//     float _binwidth;
//     float _binstart;
//     std::vector<int> _data;
// };

// template<typename Generator>
// nlohmann::json mime_bundle_repr(const histogram<Generator>& hg)
// {   g3p::gnuplot gp;
//     hg.plot(gp);
//     return mime_bundle_repr(gp);
// }

} // namespace ranx

#endif // _RANX_UTILITY_HPP_