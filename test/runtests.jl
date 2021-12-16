using Bessels
using Test
import SpecialFunctions

@time @testset "j0" begin include("j0_test.jl") end
@time @testset "y0" begin include("y0_test.jl") end

    
