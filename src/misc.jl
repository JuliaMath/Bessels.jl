using Base.Math: sin_kernel, cos_kernel, sincos_kernel, rem_pio2_kernel, DoubleFloat64, DoubleFloat32

"""
    computes sin(sum(xs)) where xs are sorted by absolute value
    Doing this is much more accurate than the naive sin(sum(xs))
"""
function sin_sum(xs::Vararg{T})::T where T<:Base.IEEEFloat
    n, y = rem_pio2_sum(xs...)
    n &= 3
    if n == 0
        return sin_kernel(y)
    elseif n == 1
        return cos_kernel(y)
    elseif n == 2
        return -sin_kernel(y)
    else
        return -cos_kernel(y)
    end
end

"""
    computes sincos(sum(xs)) where xs are sorted by absolute value
    Doing this is much more accurate than the naive sincos(sum(xs))
"""
function sincos_sum(xs::Vararg{T})::T where T<:Base.IEEEFloat
    n, y = rem_pio2_sum(xs...)
    n &= 3
    si, co = sincos_kernel(y)
    if n == 0
        return si, co
    elseif n == 1
        return co, -si
    elseif n == 2
        return -si, -co
    else
        return -co, si
    end
end

function rem_pio2_sum(xs::Vararg{Float64})
    n = 0
    hi, lo = 0.0, 0.0
    small_start = length(xs)+1
    for i in eachindex(xs)
        x = xs[i]
        if abs(x) <= pi/4
            small_start = i
            break
        end
        n_i, y = rem_pio2_kernel(x)
        n += n_i
        s = y.hi + hi
        lo += (y.hi - (s - hi)) + y.lo
        hi = s
    end
    for i in small_start:length(xs)
        x = xs[i]
        s = x + hi
        lo += (x - (s - hi))
        hi = s
    end
    while hi > pi/4
        hi -= pi/2
        lo -= 6.123233995736766e-17
        n += 1
    end
    while hi < -pi/4
        hi += pi/2
        lo += 6.123233995736766e-17
        n -= 1
    end
    return n, DoubleFloat64(hi, lo)
end

function rem_pio2_sum(xs::Vararg{Union{Float32,Float64}})
    n, y = rem_pio2_kernel(sum(Float64, xs))
    return n, DoubleFloat32(y.hi)
end
