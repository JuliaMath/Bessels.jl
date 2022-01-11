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

* it looks like SpecialFunctions.jl doesn't always preserve the correct input type so some of the calculations may be done in Float64. This might skew the benchmarks for `Bessels.jl` as it should have all calculations done in the lower precision.

```julia
# SpecialFunctions.jl 
julia> @btime besselk(1, x) setup=(x=Float32(10.0*rand()))
  179.272 ns (1 allocation: 16 bytes)
0.021948646168061196 # notice incorrect Float64 return type

# Bessels.jl
julia> @btime besselk1(x) setup=(x=Float32(10.0*rand()))
  22.967 ns (0 allocations: 0 bytes)
0.0027777348f0 # notice correct Float32 return type
```

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

```julia
# SpecialFunctions.jl 
julia> @btime besselk(1, x) setup=(x=10.0*rand())
  184.726 ns (1 allocation: 16 bytes) # notice the small difference in Float32 and Float64 implementations
0.0007517428778913419

# Bessels.jl
julia> @btime besselk1(x) setup=(x=10.0*rand())
  47.824 ns (0 allocations: 0 bytes)
0.0057366790518002045
```
