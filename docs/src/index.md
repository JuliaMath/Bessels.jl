# Bessels.jl

## Introduction

Bessels.jl is a collection of mathematical algorithms to compute special functions written entirely in the Julia programming language. It is focused on providing high quality implementations of mathematical functions that are accurate and highly performant. Since it contains only Julia code and has no external dependencies, it is a lightweight package that can be used in any high performance application while taking advatnage of the dynamism of the Julia language.

Bessels.jl contains numerous definitions of special functions important in many fields of mathematical physics. Available functions include Bessel, Modified Bessel, Spherical Bessel, Airy, Hankel, and Gamma functions. A full list of available functions can be found in the API section of this documentation. Initial development has focused on providing implementations for real arguments, however, some functions (e.g., Airy) also accept complex numbers as input.

## Scope

Bessels.jl started simply as a package to compute just the Bessel function of the first kind and zero order, but it has quickly grown to include several Bessel type functions. As mentioned earlier, Bessels.jl contains algorithms for computing Bessel, Modified Bessel, Spherical Bessel, Airy, and Hankel functions. These functions are contained in Chapters 9 and 10 of the [NIST Digital Library of Mathematical Functions](https://dlmf.nist.gov/). However, there exists other types of related functions such as the Struve and Weber type functions contained in Chapters 11 and 12. In general, these are special cases of the confuent hypergeometric functions $_0F_1, _1F_1$, and $_1F_2$. Additionally, several other special functions such as the gamma functions are also needed to compute these types of functions.

Therefore, Bessels.jl seeks to be home to any Bessel or related functions (see the section provided by [mpmath](https://mpmath.org/doc/current/functions/bessel.html)) which are typically contained in Chapters 9-12 of the NIST Digital library or to any other related function required to compute Bessel type functions.

### Advantages

- Typically 2-12x faster than other available [implementations](https://github.com/JuliaMath/SpecialFunctions.jl)
- Written entirely in Julia

### Limitations

- Does not yet support automatic differentiation
- Limited support for complex arguments
- No support for higher precision (above `Float64`)

## Packages using Bessels.jl

- [Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl)
- [GeoStats.jl](https://github.com/JuliaEarth/GeoStats.jl)
- [BesselK.jl](https://github.com/cgeoga/BesselK.jl)
- [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
- [Variography.jl](https://github.com/JuliaEarth/Variography.jl)
- [LightPropagation.jl](https://github.com/heltonmc/LightPropagation.jl)

If you are using Bessels.jl, please feel free to add yours to the list!

## Related Math libraries

**Julia packages**
- [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl)
- [HypergeometricFunctions.jl](https://github.com/JuliaMath/HypergeometricFunctions.jl)
- [ClassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl)
- [LegendrePolynomials.jl](https://github.com/jishnub/LegendrePolynomials.jl)
- [BesselK.jl](https://github.com/cgeoga/BesselK.jl)
- [AssociatedLegendrePolynomials.jl](https://github.com/jmert/AssociatedLegendrePolynomials.jl)
- [HarmonicOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/HarmonicOrthogonalPolynomials.jl)
- [SphericalHarmonics.jl](https://github.com/jishnub/SphericalHarmonics.jl)
- [ArbNumerics.jl](https://github.com/JeffreySarnoff/ArbNumerics.jl)
- [Elliptic.jl](https://github.com/nolta/Elliptic.jl)
- [EllipticFunctions.jl](https://github.com/stla/EllipticFunctions.jl)
- [ClausenFunctions.jl](https://github.com/Expander/ClausenFunctions.jl)
- [Polylogarithms.jl](https://github.com/mroughan/Polylogarithms.jl)
- [PolyLog.jl](https://github.com/Expander/PolyLog.jl)
- [LambertW.jl](https://github.com/jlapeyre/LambertW.jl)
- [Struve.jl](https://github.com/gwater/Struve.jl)
- [Quadmath.jl](https://github.com/JuliaMath/Quadmath.jl)

**Other**
- [SciPy](https://github.com/scipy/scipy)
- [mpmath](https://github.com/mpmath/mpmath)
- [Arb](https://github.com/fredrik-johansson/arb)
- [Boost](https://github.com/boostorg/boost)
- [GSL](https://www.gnu.org/software/gsl/)
- [Cephes](https://netlib.org/cephes/)
- [fortran-bessels](https://github.com/perazz/fortran-bessels)

This is by no means exhaustive so please add yours or any others that should be listed.