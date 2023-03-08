#
# SIMDMath aims to provide vectorized basic math operations to be used in static fully unrolled functions such as computing special functions.
#
# The type system is heavily influenced by SIMD.jl (https://github.com/eschnett/SIMD.jl) licensed under the Simplified "2-clause" BSD License:
# Copyright (c) 2016-2020: Erik Schnetter, Kristoffer Carlsson, Julia Computing All rights reserved.
#
# This module is also influenced by VectorizationBase.jl (https://github.com/JuliaSIMD/VectorizationBase.jl) licensed under the MIT License: Copyright (c) 2018 Chris Elrod
#

module SIMDMath

using Base: llvmcall, VecElement

export horner_simd, pack_horner
export horner, horner2, horner4, horner8
export pack_horner2, pack_horner4, pack_horner8

const VE = Base.VecElement
const FloatTypes = Union{Float16, Float32, Float64}
const LVec{N, FloatTypes} = NTuple{N, VE{FloatTypes}}

## maybe have a complex type stored as a pack or real and complex values separately
const CVec{N, FloatTypes} = NTuple{2, LVec{N, FloatTypes}}
# CVec{2, Float64}((LVec{2, Float64}((1.1, 1.2)), LVec{2, Float64}((1.5, 1.0))))

const ScalarTypes = Union{VE{FloatTypes}, FloatTypes}
const SIMDLanes = Union{LVec{2, Float64}, LVec{4, Float64}, LVec{8, Float64}}

const LLVMType = Dict{DataType, String}(
    Float16  => "half",
    Float32  => "float",
    Float64  => "double",
)

include("arithmetic.jl")
include("shufflevector.jl")
include("horner.jl")

end
