module GammaFunctions

using ..Math: SQ2PI

export gamma
export loggamma, logabsgamma, logfactorial

include("gamma.jl")
include("loggamma.jl")

end
