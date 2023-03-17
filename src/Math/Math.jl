module Math

# constains miscelaneous math functions and math constants

export cos_sum, sin_sum
export clenshaw_chebyshev

include("math_constants.jl")

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

# uses the Clenshaw algorithm to recursively evaluate a linear combination of Chebyshev polynomials
function clenshaw_chebyshev(x, c)
    x2 = 2x
    c0 = c[end-1]
    c1 = c[end]
    for i in length(c)-2:-1:1
        c0, c1 = c[i] - c1, c0 + c1 * x2
    end
    return c0 + c1 * x
end

end
