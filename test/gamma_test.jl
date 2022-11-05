for (T, max) in ((Float16, 13), (Float32, 43), (Float64, 170))
    x = rand(T, 10000)*max
    @test T.(SpecialFunctions.gamma.(widen.(x))) ≈ Bessels.gamma.(x)
    if isinteger(x)
        @test_throws DomainError Bessels.gamma.(-x)
    else
        @test T.(SpecialFunctions.gamma.(widen.(-x))) ≈ Bessels.gamma.(-x)
    end
    @test isnan(Bessels.gamma(T(NaN)))
    @test isinf(Bessels.gamma(T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ Bessels.gamma.(x)
