/*
 * Copyright (C) 2007,2008   Alex Shulgin
 *
 * This file is part of png++ the C++ wrapper for libpng.  PNG++ is free
 * software; the exact copying conditions are as follows:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
 * NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef PNGPP_RGB_PIXEL_HPP_INCLUDED
#define PNGPP_RGB_PIXEL_HPP_INCLUDED

#include "types.hpp"
#include "pixel_traits.hpp"

#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

namespace png
{

    /**
     * \brief RGB pixel type.
     */
    template< typename T >
    struct basic_rgb_pixel
    {
        /**
         * \brief Default constructor.  Initializes all components
         * with zeros.
         */
        basic_rgb_pixel()
            : red(0), green(0), blue(0)
        {
        }

        /**
         * \brief Constructs rgb_pixel object from \a red, \a green
         * and \a blue components passed as parameters.
         */
        basic_rgb_pixel(T red, T green, T blue)
            : red(red), green(green), blue(blue)
        {
        }

        T red;
        T green;
        T blue;
    };

    /**
     * The 8-bit RGB pixel type.
     */
    typedef basic_rgb_pixel< byte > rgb_pixel;

    /**
     * The 16-bit RGB pixel type.
     */
    typedef basic_rgb_pixel< uint_16 > rgb_pixel_16;

    /**
     * \brief Pixel traits specialization for basic_rgb_pixel.
     */
    template< typename T >
    struct pixel_traits< basic_rgb_pixel< T > >
        : basic_pixel_traits< basic_rgb_pixel< T >, T, color_type_rgb >
    {
    };

} // namespace png

#endif // PNGPP_RGB_PIXEL_HPP_INCLUDED

#pragma GCC diagnostic pop
