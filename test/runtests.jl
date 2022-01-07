using Bessels
using Test
import SpecialFunctions

@time @testset "besselj" begin include("besselj_test.jl") end
@time @testset "bessely" begin include("bessely_test.jl") end
