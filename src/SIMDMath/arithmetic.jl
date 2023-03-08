import Base: muladd

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