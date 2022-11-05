for (T, max) in ((Float16, 13), (Float32, 43), (Float64, 170))
    v = rand(T, 10000)*max
    for x in v
        @test T(SpecialFunctions.gamma(widen(x))) ≈ Bessels.gamma(x)
        if isinteger(x)
            @test_throws DomainError Bessels.gamma(-x)
        else
            @test isapprox(T(SpecialFunctions.gamma(widen(-x))), Bessels.gamma(-x), atol=nextfloat(Float16(0.),2))
        end
    end
    @test isnan(Bessels.gamma(T(NaN)))
    @test isinf(Bessels.gamma(T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ Bessels.gamma.(x)
