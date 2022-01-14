# Modified Bessel functions of the first kind of order zero and one
# besseli0, besseli1
# Scaled modified Bessel functions of the first kind of order zero and one
# besseli0x, besselix

#=
Approximate forms used here are given in
"Rational Approximations for the Modified Bessel Function of the First Kind - I1(x) for Computations with Double Precision"
by Pavel Holoborodko, 
http://www.advanpix.com/2015/11/12/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i1-for-computations-with-double-precision/
Actual coefficients used are from the boost math library.
https://github.com/boostorg/math/tree/develop/include/boost/math/special_functions/detail
=#
function besseli0(x::T) where T <: Union{Float32, Float64}
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
        return a * s
    end
end
function besseli0x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    x = abs(x)
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
    else
        return evalpoly(inv(x), besseli0_large_coefs(T)) / sqrt(x)
    end
end
function besseli1(x::Float32)
    T = Float32
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2
    else
        a = exp(z / 2)
        s = a * evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
        z =  a * s
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1(x::Float64)
    T = Float64
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2
    else
        a = exp(z / 2)
        s = a * evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
        z =  a * s
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::Float32)
    T = Float32
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2 * exp(-z)
    else
        z =  evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::Float64)
    T = Float64
    z = abs(x)
    if z < 7.75
        a = z * z / 4
        inner = (one(T), T(0.5), evalpoly(a, besseli1_small_coefs(T)))
        z = z * evalpoly(a, inner) / 2 * exp(-z)
    elseif z < 500
        z = evalpoly(inv(z), besseli1_med_coefs(T)) / sqrt(z)
    else
        z = evalpoly(inv(z), besseli1_large_coefs(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
