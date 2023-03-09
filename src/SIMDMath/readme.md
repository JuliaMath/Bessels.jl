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

### Case 2: Evaluating a single polynomial.

In some cases, we are interested in improving the performance when evaluating a single polynomial of larger degree. Horner's scheme is latency bound and for large polynomials (N>10) this can become a large part of the total runtime. We can test the performance of using a straight Horner scheme using the Base library function `evalpoly` against the higher order Horner schemes.
```julia
let
    horner_times = []
    horner2_times = []
    horner4_times = []
    horner8_times = []

    for N in [4, 8, 12, 16, 32, 64, 128, 256, 512]
        poly = ntuple(n -> rand()*(-1)^n / n, N)
        poly_packed2 = pack_horner2(poly)
        poly_packed4 = pack_horner4(poly)
        poly_packed8 = pack_horner8(poly)

        t1 = @benchmark evalpoly(x, $poly) setup=(x=rand())
        t2 = @benchmark horner2(x, $poly_packed2) setup=(x=rand())
        t3 = @benchmark horner4(x, $poly_packed4) setup=(x=rand())
        t4 = @benchmark horner8(x, $poly_packed8) setup=(x=rand())

        push!(horner_times,  round(minimum(t1).time, digits=3))
        push!(horner2_times,  round(minimum(t2).time, digits=3))
        push!(horner4_times,  round(minimum(t3).time, digits=3))
        push!(horner8_times,  round(minimum(t4).time, digits=3))

    end
    @show horner_times ./ horner2_times
    @show horner_times ./ horner4_times
    @show horner_times ./ horner8_times
end
```

As mentioned, Horner's scheme requires sequential multiply-add instructions that can't be performed in parallel. One way (another way is Estrin's method which we won't discuss) to improve this structure is to break the polynomial down into even and odd polynomials (a second order Horner's scheme) or into larger powers of `x^4` or `x^8` (a fourth and eighth order Horner's scheme) which allow for computing many different polynomials of similar length simultaneously. In some regard, we are just rearranging the coefficients and using the same method as we did in the first case with some additional arithmetic at the end to add all the different polynomials together.

The last fact is important because we are actually increasing the total amount of arithmetic operations needed but increasing by a large amount the number of operations that can happen in parallel. The increased operations make the advantages of this approach less straightforward than the first case which is always superior. The second and perhaps most important point is that floating point arithmetic is not associative so these approaches will give slightly different results as we are adding and multiplying in slightly differnet order.

The above benchmarks are as follows.
```julia
# Relative Speedup compared to evalpoly when evaluating a polynomial of N degree
# N = [4, 8, 12, 16, 32, 64, 128, 256, 512]

# CPU #1
Julia Version 1.8.2
Platform Info:
  OS: macOS (arm64-apple-darwin21.3.0)
  CPU: 8 × Apple M1

#   N   = [ 4,    8,    12,   16,   32,   64,   128,  256,  512]
horner2 = [1.00, 1.00, 1.16, 1.16, 2.57, 3.14,  1.59, 1.76, 2.05]
horner4 = [1.00, 0.98, 1.14, 1.00, 2.57, 4.23,  3.53, 3.87, 4.57]
horner8 = [0.88, 0.86, 1.00, 1.00, 2.29, 4.91,  6.21, 7.19, 8.45]

# CPU #2
julia> versioninfo()
Julia Version 1.8.2
Commit 36034abf260 (2022-09-29 15:21 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 12 × Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz

#   N   = [ 4,    8,    12,   16,   32,   64,   128,  256,  512]
horner2 = [1.00, 1.11, 1.30, 1.37, 2.61, 2.11, 2.09, 2.59, 2.27]
horner4 = [1.00, 0.93, 1.21, 1.44, 3.30, 5.77, 4.69, 5.41, 5.65]
horner8 = [0.68, 0.84, 0.95, 1.26, 3.05, 6.37, 6.98, 8.76, 9.45]
```
Asymptotically, we can see that the method approaches a 2, 4, and 8x increase respecitively for large degrees, however, for smaller degrees the advanges are more complex. Therefore, it is encouraged to test the performance for individual cases. Of course, this depends on statically knowing the polynomial size during compilation which allows for packing the coefficients in the most efficient way.
