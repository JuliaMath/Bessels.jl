module Bessels

using .AiryFunctions

export besselj0
export besselj1
export besselj

export bessely0
export bessely1
export bessely

export sphericalbesselj
export sphericalbessely

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

const ComplexOrReal{T} = Union{T,Complex{T}}

include("besseli.jl")
include("besselj.jl")
include("besselk.jl")
include("bessely.jl")
include("hankel.jl")
include("sphericalbessel.jl")
include("modifiedsphericalbessel.jl")

include("SIMDMath/SIMDMath.jl")
include("Math/Math.jl")

include("AiryFunctions/AiryFunctions.jl")

include("Float128/besseli.jl")
include("Float128/besselj.jl")
include("Float128/besselk.jl")
include("Float128/bessely.jl")
include("Float128/constants.jl")

include("constants.jl")
include("U_polynomials.jl")
include("recurrence.jl")
include("Polynomials/besselj_polys.jl")
include("asymptotics.jl")
include("gamma.jl")

precompile(besselj, (Float64, Float64))

end
