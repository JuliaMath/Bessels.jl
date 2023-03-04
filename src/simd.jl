
module HornerSIMD
# inspiration from SIMD.jl and VectorizationBase.jl

using Base: llvmcall, VecElement
import Base: muladd

export evalpoly_packed

const VE = Base.VecElement
const FloatTypes = Union{Float32, Float64}
const LVec{N, FloatTypes} = NTuple{N, VE{FloatTypes}}
const ScalarTypes = Union{VE{FloatTypes}, FloatTypes}
const SIMDLanes = Union{LVec{2, Float64}, LVec{4, Float64}, LVec{8, Float64}}


# create tuples of VecElements filled with a constant value of x

@inline constantvector(x::Float64, y) = constantvector(VE(x), y)

@inline function constantvector(x::VecElement{Float64}, y::Type{LVec{2, Float64}})
    llvmcall("""%2 = insertelement <2 x double> undef, double %0, i32 0
                %3 = shufflevector <2 x double> %2, <2 x double> undef, <2 x i32> zeroinitializer
                ret <2 x double> %3""",
        LVec{2, Float64},
        Tuple{VecElement{Float64}},
        x)
end
@inline function constantvector(x::VecElement{Float64}, y::Type{LVec{4, Float64}})
    llvmcall("""%2 = insertelement <4 x double> undef, double %0, i32 0
                %3 = shufflevector <4 x double> %2, <4 x double> undef, <4 x i32> zeroinitializer
                ret <4 x double> %3""",
        LVec{4, Float64},
        Tuple{VecElement{Float64}},
        x)
end
@inline function constantvector(x::VecElement{Float64}, y::Type{LVec{8, Float64}})
    llvmcall("""%2 = insertelement <8 x double> undef, double %0, i32 0
                %3 = shufflevector <8 x double> %2, <8 x double> undef, <8 x i32> zeroinitializer
                ret <8 x double> %3""",
        LVec{8, Float64},
        Tuple{VecElement{Float64}},
        x)
end

# promotions for scalar variables to LVec
#### need to think about the type being just a Float or a single VecElement will need to widen the Float Types to include a single VecElement

@inline muladd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes} = _muladd(x, y, z)
@inline muladd(x::ScalarTypes, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), y, z)
@inline muladd(x::LVec{N, T}, y::ScalarTypes, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(x, constantvector(y, LVec{N, T}), z)
@inline muladd(x::ScalarTypes, y::ScalarTypes, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), constantvector(y, LVec{N, T}), z)
@inline muladd(x::LVec{N, T}, y::LVec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(x, y, constantvector(z, LVec{N, T}))
@inline muladd(x::ScalarTypes, y::LVec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), y, constantvector(z, LVec{N, T}))
@inline muladd(x::LVec{N, T}, y::ScalarTypes, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(x, constantvector(y, LVec{N, T}), constantvector(z, LVec{N, T}))

# muladd llvm instructions

@inline function _muladd(x::LVec{2, Float64}, y::LVec{2, Float64}, z::LVec{2, Float64})
    llvmcall("""%4 = fmul contract <2 x double> %0, %1
                %5 = fadd contract <2 x double> %4, %2
                ret <2 x double> %5""",
        LVec{2, Float64},
        Tuple{LVec{2, Float64}, LVec{2, Float64}, LVec{2, Float64}},
        x,
        y,
        z)
end

@inline function _muladd(x::LVec{4, Float64}, y::LVec{4, Float64}, z::LVec{4, Float64})
    llvmcall("""%4 = fmul contract <4 x double> %0, %1
                %5 = fadd contract <4 x double> %4, %2
                ret <4 x double> %5""",
        LVec{4, Float64},
        Tuple{LVec{4, Float64}, LVec{4, Float64}, LVec{4, Float64}},
        x,
        y,
        z)
end

@inline function _muladd(x::LVec{8, Float64}, y::LVec{8, Float64}, z::LVec{8, Float64})
    llvmcall("""%4 = fmul contract <8 x double> %0, %1
                %5 = fadd contract <8 x double> %4, %2
                ret <8 x double> %5""",
        LVec{8, Float64},
        Tuple{LVec{8, Float64}, LVec{8, Float64}, LVec{8, Float64}},
        x,
        y,
        z)
end

evalpoly_packed(x::Union{T, VE{T}}, p::NTuple{N, LVec{M, T}}) where {N, M, T} = evalpoly_packed(constantvector(x, LVec{M, T}), p)

# the type needs to only accept a custom type for LVec{2, Float64}, LVec{4, Float64}....
@inline function evalpoly_packed(x::LVec{M, Float64}, p::NTuple{N, LVec{M, Float64}}) where {N, M}
    a = p[end]
    @inbounds for i in N-1:-1:1
        a = muladd(x, a, p[i])
    end
    return a
end

end
