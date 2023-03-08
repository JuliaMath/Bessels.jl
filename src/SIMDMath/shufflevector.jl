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
