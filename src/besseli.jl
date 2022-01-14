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
    T == Float32 ? branch = 50 : branch = 500
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, P1_i0(T)), 1)
    elseif x < branch
        return exp(x) * evalpoly(inv(x), P2_i0(T)) / sqrt(x)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), P3_i0(T)) / sqrt(x)
        return a * s
    end
end
function besseli0x(x::T) where T <: Union{Float32, Float64}
    T == Float32 ? branch = 50 : branch = 500
    if x < 7.75
        a = x * x / 4
        return muladd(a, evalpoly(a, P1_i0(T)), 1) * exp(-x)
    elseif x < branch
        return evalpoly(inv(x), P2_i0(T)) / sqrt(x)
    else
        return evalpoly(inv(x), P3_i0(T)) / sqrt(x)
    end
end
function besseli1(x::Float32)
    T = Float32
    if x < 7.75
        a = x * x / 4
        inner = (one(T), T(0.5), evalpoly(a, P1_i1(T)))
        return x * evalpoly(a, inner) / 2
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), P2_i1(T)) / sqrt(x)
        return a * s
    end
end
function besseli1(x::Float64)
    T = Float64
    if x < 7.75
        a = x * x / 4
        inner = (one(T), T(0.5), evalpoly(a, P1_i1(T)))
        return x * evalpoly(a, inner) / 2
    elseif x < 500
        return exp(x) * evalpoly(inv(x), P2_i1(T)) / sqrt(x)
    else
        a = exp(x / 2)
        s = a * evalpoly(inv(x), P3_i1(T)) / sqrt(x)
        return a * s
    end
end
function besseli1x(x::Float32)
    T = Float32
    if x < 7.75
        a = x * x / 4
        inner = (one(T), T(0.5), evalpoly(a, P1_i1(T)))
        return x * evalpoly(a, inner) / 2 * exp(-x)
    else
        return evalpoly(inv(x), P2_i1(T)) / sqrt(x)
    end
end
function besseli1x(x::Float64)
    T = Float64
    if x < 7.75
        a = x * x / 4
        inner = (one(T), T(0.5), evalpoly(a, P1_i1(T)))
        return x * evalpoly(a, inner) / 2 * exp(-x)
    elseif x < 500
        return evalpoly(inv(x), P2_i1(T)) / sqrt(x)
    else
        return evalpoly(inv(x), P3_i1(T)) / sqrt(x)
    end
end
