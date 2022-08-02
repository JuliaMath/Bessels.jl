using Bessels
using Test
import SpecialFunctions

@time @testset "besseli" begin include("besseli_test.jl") end
@time @testset "besselk" begin include("besselk_test.jl") end
@time @testset "besselj" begin include("besselj_test.jl") end
@time @testset "bessely" begin include("bessely_test.jl") end
@time @testset "gamma" begin include("gamma_test.jl") end
