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

function simd_evalpoly_width4(μ, ps::NTuple{N, NTuple{4, T}}) where {N, T}
    s1 = ps[N][1]
    s2 = ps[N][2]
    s3 = ps[N][3]
    s4 = ps[N][4]
    @inbounds for i in N-1:-1:1
        s1 = muladd(μ, s1, ps[i][1])
        s2 = muladd(μ, s2, ps[i][2])
        s3 = muladd(μ, s3, ps[i][3])
        s4 = muladd(μ, s2, ps[i][4])
    end
    return (s1, s2, s3, s4)
end
