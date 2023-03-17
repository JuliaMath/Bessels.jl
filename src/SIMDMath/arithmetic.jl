import Base: muladd

## ShuffleVector

# create tuples of VecElements filled with a constant value of x
@inline constantvector(x::T, y) where T <: FloatTypes = constantvector(VE(x), y)

@inline @generated function constantvector(x::VecElement{T}, y::Type{LVec{N, T}}) where {N, T <: FloatTypes}
    s = """
        %2 = insertelement <$N x $(LLVMType[T])> undef, $(LLVMType[T]) %0, i32 0
        %3 = shufflevector <$N x $(LLVMType[T])> %2, <$N x $(LLVMType[T])> undef, <$N x i32> zeroinitializer
        ret <$N x $(LLVMType[T])> %3
        """
    return :(
        llvmcall($s, LVec{N, T}, Tuple{VecElement{T}}, x)
        )
end

## muladd

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

for f in (:fadd, :fsub, :fmul, :fdiv)
    @eval @inline @generated function $f(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: FloatTypes}
        ff = $(QuoteNode(f))
        s = """
        %3 = $ff <$N x $(LLVMType[T])> %0, %1
        ret <$N x $(LLVMType[T])> %3
        """
        return :(
        llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
    end
end
