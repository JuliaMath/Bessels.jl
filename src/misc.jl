# function to more accurately compute cos(x + xn)
# see https://github.com/heltonmc/Bessels.jl/pull/13
# written by @oscardssmith
function cos_sum(x, xn)
    s = x + xn
    n, r = reduce_pi02_med(s)
    lo = r.lo - ((s - x) - xn)
    hi = r.hi + lo
    y = Base.Math.DoubleFloat64(hi, r.hi-hi+lo)
    n = n&3
    if n == 0
        return Base.Math.cos_kernel(y)
    elseif n == 1
        return -Base.Math.sin_kernel(y)
    elseif n == 2
        return -Base.Math.cos_kernel(y)
    else
        return Base.Math.sin_kernel(y)
    end
end
# function to more accurately compute sin(x + xn)
function sin_sum(x, xn)
    s = x + xn
    n, r = reduce_pi02_med(s)
    lo = r.lo - ((s - x) - xn)
    hi = r.hi + lo
    y = Base.Math.DoubleFloat64(hi, r.hi-hi+lo)
    n = n&3
    if n == 0
        return Base.Math.sin_kernel(y)
    elseif n == 1
        return Base.Math.cos_kernel(y)
    elseif n == 2
        return -Base.Math.sin_kernel(y)
    else
        return -Base.Math.cos_kernel(y)
    end
end
@inline function reduce_pi02_med(x::Float64)
    pio2_1 = 1.57079632673412561417e+00

    fn = round(x*(2/pi))
    r  = muladd(-fn, pio2_1, x)
    w  = fn * 6.07710050650619224932e-11
    y = r-w
    return unsafe_trunc(Int, fn), Base.Math.DoubleFloat64(y, (r-y)-w)
end

# performs a fourth order horner scheme for polynomial evaluation
# computes the even and odd coefficients of the polynomial independently within a loop to reduce latency
# splits the polynomial to compute both 1 + ax + bx^2 + cx^3 and 1 - ax + bx^2 - cx^3 ....
# return both 1 + ax + bx^2 + cx^3 and 1 - ax + bx^2 - cx^3 ....
# uses a fourth order Horner scheme
@inline function horner_split(x, P)
    x2 = x * x
    x4 = x2 * x2
    p = horner_simd(x4, P)
    a0 = SIMDMath.Vec(SIMDMath.shufflevector(p.data, Val(0:1)))
    b0 = SIMDMath.Vec(SIMDMath.shufflevector(p.data, Val(2:3)))
    p1 = horner_simd(x2, (a0, b0))
    a1 = SIMDMath.Vec(SIMDMath.shufflevector(p1.data, Val(0)))
    b1 = SIMDMath.Vec(SIMDMath.shufflevector(p1.data, Val(1)))
    return SIMDMath.fmadd(-x, b1, a1).data[1].value,  SIMDMath.fmadd(x, b1, a1).data[1].value
end
