module Bessels

export besselj0
export besselj1
export besselj

export bessely0
export bessely1
export bessely

export sphericalbesselj
export sphericalbessely
export sphericalbesseli
export sphericalbesselk

export besseli
export besselix
export besseli0
export besseli0x
export besseli1
export besseli1x

export besselk
export besselkx
export besselk0
export besselk0x
export besselk1
export besselk1x

export besselh
export hankelh1
export hankelh2

export airyai
export airyaix
export airyaiprime
export airyaiprimex
export airybi
export airybix
export airybiprime
export airybiprimex

export gamma

const ComplexOrReal{T} = Union{T,Complex{T}}

include("SIMDMath/SIMDMath.jl")
include("Math/Math.jl")
include("GammaFunctions/GammaFunctions.jl")
include("BesselFunctions/BesselFunctions.jl")
include("AiryFunctions/AiryFunctions.jl")

using .GammaFunctions
using .BesselFunctions
using .AiryFunctions

precompile(besselj, (Float64, Float64))

end
