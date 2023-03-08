# SIMDMath.jl

A lightweight module for explicit vectorization of simple math functions. The focus is mainly on vectorizing polynomial evaluation in two main cases: (1) evaluating many different polynomials of similar length and (2) evaluating a single large polynomial.
This module is for statically known functions where the coefficients are unrolled and the size of the tuples is known at compilation. For more advanced needs it will be better to use SIMD.jl or LoopVectorization.jl.

### Case 1: Evaluating many different polynomials.

In the evaluation of special functions, we often need to compute many polynomials at the same `x`. An example structure would look like...
```julia
const NT = 12
const P = (
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT),
           ntuple(n -> rand()*(-1)^n / n, NT)
       )

function test(x)
           x2 = x * x
           x3 = x2 * x

           p1 = evalpoly(x3, P[1])
           p2 = evalpoly(x3, P[2])
           p3 = evalpoly(x3, P[3])
           p4 = evalpoly(x3, P[4])

           return muladd(x, -p2, p1), muladd(x2, p4, -p3)
       end
```
This structure is advantageous for vectorizing as `p1`, `p2`, `p3`, and `p4` are independent, require same number of evaluations, and coefficients are statically known.
However, we are relying on the auto-vectorizer to make sure this happens which is very fragile. In general, two polynomials might auto-vectorizer depending on how the values are used but is not reliable.
We can check that this function is not vectorizing (though it may on some architectures) by using `@code_llvm test(1.1)` and/or `@code_native(1.1)`.

Another way to test this is to benchmark this function and compare to the time to compute a single polynomial.
```julia
julia> @btime test(x) setup=(x=rand()*2)
  13.026 ns (0 allocations: 0 bytes)

julia> @btime evalpoly(x, P[1]) setup=(x=rand()*2)
  3.973 ns (0 allocations: 0 bytes)
```
In this case, `test` is almost 4x longer as all the polynomial evaluations are happening sequentially.

We can do much better by making sure these polynomials vectorize.
```julia
# using the same coefficients as above
julia> using Bessels.SIMDMath

@inline function test_simd(x)
       x2 = x * x
       x3 = x2 * x
       p = horner_simd(x3, pack_horner(P...))
       return muladd(x, -p[2].value, p[1].value), muladd(x2, p[4].value, -p[3].value)
end

julia> @btime test_simd(x) setup=(x=rand()*2)
  4.440 ns (0 allocations: 0 bytes)
```


