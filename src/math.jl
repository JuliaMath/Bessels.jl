using Base.Math: sin_kernel, cos_kernel, sincos_kernel, rem_pio2_kernel, DoubleFloat64, DoubleFloat32

"""
    computes sin(sum(xs)) where xs are sorted by absolute value
    Doing this is much more accurate than the naive sin(sum(xs))
"""
function sin_sum(xs::Vararg{T, N})::T where {T<:Base.IEEEFloat, N}
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
function sincos_sum(xs::Vararg{T, N})::T where {T<:Base.IEEEFloat, N}
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

function rem_pio2_sum(xs::Vararg{Float64, N}) where N
    n = 0
    hi, lo = 0.0, 0.0
    for x in xs
        if abs(x) <= pi/4
            s = x + hi
            lo += (x - (s - hi))
        else
            n_i, y = rem_pio2_kernel(x)
            n += n_i
            s = y.hi + hi
            lo += (y.hi - (s - hi)) + y.lo
        end
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

function rem_pio2_sum(xs::Vararg{Float32, N}) where N
    y = 0.0
    n = 0
    # The minimum cosine or sine of any Float32 that gets reduced is 1.6e-9
    # so reducing at 2^22 prevents catastrophic loss of precision.
    # There probably is a case where this loses some digits but it is a decent
    # tradeoff between accuracy and speed.
    @fastmath for x in xs
        if x > 0x1p22
            n_i, y_i = rem_pio2_kernel(Float32(x))
            n += n_i
            y += y_i.hi
        else
            y += x
        end
    end
    n_i, y = rem_pio2_kernel(y)
    return n + n_i, DoubleFloat32(y.hi)
end

function rem_pio2_sum(xs::Vararg{Float16, N}) where N
    y = sum(Float64, xs) #Float16 can be losslessly accumulated in Float64
    n, y = rem_pio2_kernel(y)
    return n, DoubleFloat32(y.hi)
end

# Levin's Sequence transformation

using Base.Cartesian
using SIMDMath: fmadd, Vec, FloatTypes

#@inline levin_scale(B::T, n, k) where T = -(B + n) * (B + n + k)^(k - one(T)) / (B + n + k + one(T))^k
@inline levin_scale(B::T, n, k) where T = -(B + n + k) * (B + n + k - 1) / ((B + n + 2k) * (B + n + 2k - 1))

# implementation for real numbers
@inline @generated function levin_transform(s::NTuple{N, Vec{2, T}}) where {N, T <: FloatTypes}
    len = N - 1
    :(
        begin
            @nexprs $N i -> a_{i} = s[i]
            @nexprs $len k -> (@nexprs ($len-k) i -> a_{i} = fmadd(a_{i}, levin_scale(one(T), i, k-1), a_{i+1}))
            return (a_1[1] / a_1[2])
        end
    )
end

# implementation for complex numbers
@inline @generated function levin_transform(s::NTuple{N, Vec{4, T}}) where {N, T <: FloatTypes}
    len = N - 1
    :(
        begin
            @nexprs $N i -> a_{i} = s[i]
            @nexprs $len k -> (@nexprs ($len-k) i -> a_{i} = fmadd(a_{i}, levin_scale(one(T), i, k-1), a_{i+1}))
            return (complex(a_1[1], a_1[2]) / complex(a_1[3], a_1[4]))
        end
    )
end
