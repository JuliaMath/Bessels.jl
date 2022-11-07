for (T, max, rtol) in ((Float16, 13, 1.0), (Float32, 43, 1.0), (Float64, 170, 7))
    v = rand(T, 10000)*max
    for x in v
        @test isapprox(T(SpecialFunctions.gamma(widen(x))), Bessels.gamma(x), rtol=rtol*eps(T))
        if isinteger(x)
            @test_throws DomainError Bessels.gamma(-x)
        else
            @test isapprox(T(SpecialFunctions.gamma(widen(-x))), Bessels.gamma(-x), atol=nextfloat(T(0.),2), rtol=rtol*eps(T))
        end
    end
    @test isnan(Bessels.gamma(T(NaN)))
    @test isinf(Bessels.gamma(T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ Bessels.gamma.(x)
