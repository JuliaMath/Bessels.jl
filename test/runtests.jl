using Bessels
using Test
import SpecialFunctions

@time @testset "besseli" begin include("besseli_test.jl") end
@time @testset "besselk" begin include("besselk_test.jl") end
@time @testset "besselj" begin include("besselj_test.jl") end
@time @testset "bessely" begin include("bessely_test.jl") end
@time @testset "hankel" begin include("hankel_test.jl") end
@time @testset "gamma" begin include("gamma_test.jl") end
@time @testset "airy" begin include("airy_test.jl") end
@time @testset "sphericalbessel" begin include("sphericalbessel_test.jl") end
@time @testset "besselk enzyme autodiff" begin include("besselk_enzyme_test.jl") end
@time @testset "besseli enzyme autodiff" begin include("besseli_enzyme_test.jl") end
