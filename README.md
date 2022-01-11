# Bessels.jl

Implementations of Bessel's functions `besselj0`, `besselj1`, `bessely0`, `bessely1`, `besseli0`, `besseli1`, `besselk0`, `besselk1` in Julia.
Most implementations are ported to Julia from the [Cephes](https://www.netlib.org/cephes/) math library  originally written by Stephen L. Moshier in the C programming language
which are (partly) used in [SciPy](https://docs.scipy.org/doc/scipy/reference/special.html#faster-versions-of-common-bessel-functions) and [GCC libquadmath](https://gcc.gnu.org/onlinedocs/libquadmath/).

# Benchmarks

Comparing the relative speed (`SpecialFunctions.jl / Bessels.jl`) for a vector of values between 0 and 100.

## Float32

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.7x |
| bessely0  | 1.6x  |
| bessely1  | 1.6x  |
| besseli0  | 24x  |
| besseli1  | 22x  |
| besselk0  | 13x  |
| besselk1  | 15x  |

## Float64

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.9x |
| bessely0  | 1.5x  |
| bessely1  | 1.5x  |
| besseli0  | 9x  |
| besseli1  | 7x  |
| besselk0  | 5x  |
| besselk1  | 5x  |
