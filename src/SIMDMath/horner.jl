
# promotions for scalar variables to LVec
#### need to think about the type being just a Float or a single VecElement will need to widen the Float Types to include a single VecElement

evalpoly_packed(x::Union{T, VE{T}}, p::NTuple{N, LVec{M, T}}) where {N, M, T} = evalpoly_packed(constantvector(x, LVec{M, T}), p)

# the type needs to only accept a custom type for LVec{2, Float64}, LVec{4, Float64}....
@inline function evalpoly_packed(x::LVec{M, Float64}, p::NTuple{N, LVec{M, Float64}}) where {N, M}
    a = p[end]
    @inbounds for i in N-1:-1:1
        a = muladd(x, a, p[i])
    end
    return a
end

@inline pack_poly(p1::NTuple{N, T}, p2::NTuple{N, T}, p3::NTuple{N, T}, p4::NTuple{N, T}) where {N, T <: Float64} = ntuple(i -> LVec{4, T}((p1[i], p2[i], p3[i], p4[i])), Val(N))



# performs a second order horner scheme that splits polynomial evaluation into even/odd powers
@inline horner2(x, P::NTuple{N, Float64}) where N = horner2(x, pack_poly2(P))

@inline function horner2(x, P::NTuple{N, LVec{2, Float64}}) where N
    a = evalpoly_packed(x * x, P)
    return muladd(x, a[2].value, a[1].value)
end

# do we try to vectorize the final line as the two muladds can be done in parallel
# they are in reasonable order it would be about splitting the a1, a2, a3, a4 into a two by two 
# we can then do the same for the 8th order scheme
## 4th order horner scheme that splits into a + ex^4 + ... & b + fx^5 ... & etc
@inline horner4(x, P::NTuple{N, Float64}) where N = horner4(x, pack_poly4(P))

@inline function horner4(x, P::NTuple{N, LVec{4, Float64}}) where N
    xx = x * x
    a = evalpoly_packed(xx * xx, P)
    b = muladd(x, LVec{2, Float64}((a[4].value, a[2].value)), LVec{2, Float64}((a[3].value, a[1].value)))
    return muladd(xx, b[1].value, b[2].value) 
end

# 8th order horner scheme that splits into a + ix^8 + .... & etc

@inline horner8(x, P::NTuple{N, Float64}) where N = horner8(x, pack_poly8(P))

@inline function horner8(x, P::NTuple{N, LVec{8, Float64}}) where N
    x2 = x * x
    x4 = x2 * x2
    a = evalpoly_packed(x4 * x4, P)

    # following computes
    # a[1].value + a[2].value*x + a[3].value*x^2 + a[4].value*x^3 + a[5].value*x^4 + a[6].value*x^5 + a[7].value*x^6 + a[8].value*x^7

    b = muladd(x, LVec{4, Float64}((a[4].value, a[2].value, a[8].value, a[6].value)), LVec{4, Float64}((a[3].value, a[1].value, a[7].value, a[5].value)))
    c = muladd(x2, LVec{2, Float64}((b[1].value, b[3].value)), LVec{2, Float64}((b[2].value, b[4].value)))
    return muladd(x4, c[2].value, c[1].value)
end

@inline function pack_poly2(p::NTuple{N, T}) where {N, T <: Float64}
    isone(N) && return LVec{2, T}((p[1], 0.0))
    if iseven(N)
        return ntuple(i -> LVec{2, T}((p[2*i - 1], p[2i])), Val(N ÷ 2))
    else
        return ntuple(i -> (isequal(i, (N + 1) ÷ 2) ? (LVec{2, T}((p[2*i - 1], 0.0))) :  LVec{2, T}((p[2*i - 1], p[2i]))), Val((N + 1) ÷ 2))
    end
end

@inline function pack_poly4(p::NTuple{N, T}) where {N, T <: Float64}
    rem = N % 4
    if !iszero(rem)
        P = (p..., ntuple(i -> 0.0, Val(4 - rem))...)
        return ntuple(i -> LVec{4, T}((P[4*i - 3], P[4i - 2], P[4i - 1], P[4i])), Val(length(P) ÷ 4))
    else
        return ntuple(i -> LVec{4, T}((p[4*i - 3], p[4i - 2], p[4i - 1], p[4i])), Val(N ÷ 4))
    end
end

@inline function pack_poly8(p::NTuple{N, T}) where {N, T <: Float64}
    rem = N % 8
    if !iszero(rem)
        P = (p..., ntuple(i -> 0.0, Val(8 - rem))...)
        return ntuple(i -> LVec{8, T}((P[8i - 7], P[8i - 6], P[8i - 5], P[8i - 4], P[8i - 3], P[8i - 2], P[8i - 1], P[8i])), Val(length(P) ÷ 8))
    else
        return ntuple(i -> LVec{8, T}((p[8i - 7], p[8i - 6], p[8i - 5], p[8i - 4], p[8i - 3], p[8i - 2], p[8i - 1], p[8i])), Val(N ÷ 8))
    end
end
