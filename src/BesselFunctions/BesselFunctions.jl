module BesselFunctions

using ..Bessels: ComplexOrReal
using ..GammaFunctions
using ..Math

export besselj0
export besselj1
export besselj
export besselj!

export bessely0
export bessely1
export bessely
export bessely!

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
export besseli!

export besselk
export besselkx
export besselk0
export besselk0x
export besselk1
export besselk1x
export besselk!

export besselh
export hankelh1
export hankelh2

include("besseli.jl")
include("besselj.jl")
include("besselk.jl")
include("bessely.jl")
include("hankel.jl")
include("sphericalbessel.jl")
include("modifiedsphericalbessel.jl")


include("constants.jl")
include("U_polynomials.jl")
include("recurrence.jl")
include("Polynomials/besselj_polys.jl")
include("asymptotics.jl")

include("Float128/constants.jl")
include("Float128/besselj.jl")
include("Float128/bessely.jl")

end
