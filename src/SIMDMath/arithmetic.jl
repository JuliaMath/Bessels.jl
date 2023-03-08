import Base: muladd

@inline muladd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes} = _muladd(x, y, z)
@inline muladd(x::ScalarTypes, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), y, z)
@inline muladd(x::LVec{N, T}, y::ScalarTypes, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(x, constantvector(y, LVec{N, T}), z)
@inline muladd(x::ScalarTypes, y::ScalarTypes, z::LVec{N, T}) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), constantvector(y, LVec{N, T}), z)
@inline muladd(x::LVec{N, T}, y::LVec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(x, y, constantvector(z, LVec{N, T}))
@inline muladd(x::ScalarTypes, y::LVec{N, T}, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(constantvector(x, LVec{N, T}), y, constantvector(z, LVec{N, T}))
@inline muladd(x::LVec{N, T}, y::ScalarTypes, z::ScalarTypes) where {N, T <: FloatTypes} = muladd(x, constantvector(y, LVec{N, T}), constantvector(z, LVec{N, T}))

# muladd llvm instructions

@inline @generated function _muladd(x::LVec{N, T}, y::LVec{N, T}, z::LVec{N, T}) where {N, T <: FloatTypes}
    s = """
        %4 = fmul contract <$N x $(LLVMType[T])> %0, %1
        %5 = fadd contract <$N x $(LLVMType[T])> %4, %2
        ret <$N x $(LLVMType[T])> %5
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}, LVec{N, T}}, x, y, z)
        )
end
