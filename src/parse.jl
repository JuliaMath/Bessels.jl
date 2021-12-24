@inline function Float128(str::S) where {S<:AbstractString}
    return Float128(BigFloat(str))
end
  
macro f128_str(val::AbstractString)
    :(Float128($val))
end