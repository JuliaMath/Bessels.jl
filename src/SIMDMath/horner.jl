#
# Polynomial evaluation using Horner's scheme.
#
# File contains methods to compute several different polynomials at once using SIMD with Horner's scheme (horner_simd).
# Additional methods to evaluate a single polynomial using SIMD with different order Horner's scheme (horner2, horner4, horner8).
# TODO: Extend to Float32
# TODO: Support complex numbers

#
# Parallel Horner's scheme to evaluate 2, 4, or 8 polynomials in parallel using SIMD.
#
# Usage:
# x = 1.1
# P1 = (1.1, 1.2, 1.4, 1.5)
# P2 = (1.3, 1.3, 1.6, 1.8)
# a = horner_simd(x, pack_horner(P1, P2))
# a[1].value == evalpoly(x, P1)
# a[2].value == evalpoly(x, P2)
#
# Note the strict equality as this method doesn't alter the order of individual polynomial evaluations
#

# cast single element `x` to a width of the number of polynomials
horner_simd(x::Union{T, VE{T}}, p::NTuple{N, LVec{M, T}}) where {N, M, T} = horner_simd(constantvector(x, LVec{M, T}), p)

# In theory, this can accept any number of packed polynomials though for SIMD this really should only accept `M = 2, 4, 8` and for Float32 `M = 16`
@inline function horner_simd(x::LVec{M, Float64}, p::NTuple{N, LVec{M, Float64}}) where {N, M}
    a = p[end]
    @inbounds for i in N-1:-1:1
        a = muladd(x, a, p[i])
    end
    return a
end

@inline pack_horner(p1::NTuple{N, T}, p2::NTuple{N, T}) where {N, T <: Float64} = ntuple(i -> LVec{2, T}((p1[i], p2[i])), Val(N))
@inline pack_horner(p1::NTuple{N, T}, p2::NTuple{N, T}, p3::NTuple{N, T}, p4::NTuple{N, T}) where {N, T <: Float64} = ntuple(i -> LVec{4, T}((p1[i], p2[i], p3[i], p4[i])), Val(N))
@inline pack_horner(p1::NTuple{N, T}, p2::NTuple{N, T}, p3::NTuple{N, T}, p4::NTuple{N, T}, p5::NTuple{N, T}, p6::NTuple{N, T}, p7::NTuple{N, T}, p8::NTuple{N, T}) where {N, T <: Float64} = ntuple(i -> LVec{8, T}((p1[i], p2[i], p3[i], p4[i], p5[i], p6[i], p7[i], p8[i])), Val(N))

#
# Horner scheme for evaluating single polynomials
#
# In some cases we are interested in speeding up the evaluation of a single polynomial of high degree.
# It is possible to split the polynomial into pieces and use SIMD to convert each piece simultaneously
#
# Usage:
# x = 1.1
# poly = (1.0, 0.5, 0.1, 0.05, 0.1)
# evalpoly(x, poly) ≈ horner2(x, pack_horner2(poly)) ≈ horner4(x, pack_horner4(poly)) ≈ horner8(x, pack_horner8(poly))
#
# Note the approximative relation between each as floating point arithmetic is associative.
# Here, we are adding up the polynomial in different degrees so there will be a few ULPs difference
# 

# horner - default regular horner to base evalpoly
# horner2 - 2nd horner scheme that splits polynomial into even/odd powers
# horner4 - 4th order horner scheme that splits into a + ex^4 + ... & b + fx^5 ... & etc
# horner8 - 8th order horner scheme that splits into a + ix^8 + .... & etc

@inline horner(x::T, P::NTuple{N, T}) where {N, T <: FloatTypes} = evalpoly(x, P)
@inline horner2(x, P::NTuple{N, Float64}) where N = horner2(x, pack_horner2(P))
@inline horner4(x, P::NTuple{N, Float64}) where N = horner4(x, pack_horner4(P))
@inline horner8(x, P::NTuple{N, Float64}) where N = horner8(x, pack_horner8(P))

@inline function horner2(x, P::NTuple{N, LVec{2, Float64}}) where N
    a = horner_simd(x * x, P)
    return muladd(x, a[2].value, a[1].value)
end

@inline function horner4(x, P::NTuple{N, LVec{4, Float64}}) where N
    xx = x * x
    a = horner_simd(xx * xx, P)
    b = muladd(x, LVec{2, Float64}((a[4].value, a[2].value)), LVec{2, Float64}((a[3].value, a[1].value)))
    return muladd(xx, b[1].value, b[2].value) 
end

@inline function horner8(x, P::NTuple{N, LVec{8, Float64}}) where N
    x2 = x * x
    x4 = x2 * x2
    a = horner_simd(x4 * x4, P)

    # following computes
    # a[1].value + a[2].value*x + a[3].value*x^2 + a[4].value*x^3 + a[5].value*x^4 + a[6].value*x^5 + a[7].value*x^6 + a[8].value*x^7

    b = muladd(x, LVec{4, Float64}((a[4].value, a[2].value, a[8].value, a[6].value)), LVec{4, Float64}((a[3].value, a[1].value, a[7].value, a[5].value)))
    c = muladd(x2, LVec{2, Float64}((b[1].value, b[3].value)), LVec{2, Float64}((b[2].value, b[4].value)))
    return muladd(x4, c[2].value, c[1].value)
end

#
# Following packs coefficients for second, fourth, and eighth order horner evaluations.
# Accepts a single polynomial that will pack into proper coefficients
#
# Usage:
# x = 1.1
# poly = (1.0, 0.5, 0.1, 0.05, 0.1)
# evalpoly(x, poly) ≈ horner2(x, pack_horner2(poly)) ≈ horner4(x, pack_horner4(poly)) ≈ horner8(x, pack_horner8(poly))
#
# Note: these functions will pack statically known polynomials for N <= 32 at compile time.
# If length(poly) > 32 they must be packed and declared as constants otherwise packing will allocate at runtime
# Example:
# const P = pack_horner2(ntuple(i -> i*0.1, 35))
# horner2(x, P)
#
# TODO: Use a generated function instead might make the packing more reliable for all polynomial degrees
#

@inline function pack_horner2(p::NTuple{N, T}) where {N, T <: Float64}
    rem = N % 2
    pad = !iszero(rem) ? (2 - rem) : 0
    P = (p..., ntuple(i -> 0.0, Val(pad))...)
    return ntuple(i -> LVec{2, T}((P[2*i - 1], P[2i])), Val((N + pad) ÷ 2))
end

@inline function pack_horner4(p::NTuple{N, T}) where {N, T <: Float64}
    rem = N % 4
    pad = !iszero(rem) ? (4 - rem) : 0
    P = (p..., ntuple(i -> 0.0, Val(pad))...)
    return ntuple(i -> LVec{4, T}((P[4*i - 3], P[4i - 2], P[4i - 1], P[4i])), Val((N + pad) ÷ 4))
end

@inline function pack_horner8(p::NTuple{N, T}) where {N, T <: Float64}
    rem = N % 8
    pad = !iszero(rem) ? (8 - rem) : 0
    P = (p..., ntuple(i -> 0.0, Val(pad))...)
    return ntuple(i -> LVec{8, T}((P[8i - 7], P[8i - 6], P[8i - 5], P[8i - 4], P[8i - 3], P[8i - 2], P[8i - 1], P[8i])), Val((N + pad) ÷ 8))
end
