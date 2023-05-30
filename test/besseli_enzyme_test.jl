using EnzymeCore, Enzyme

dbesseli_dv(v, x) = autodiff(Forward, _v->besseli(_v, x), 
                             Duplicated, Duplicated(v, 1.0))[2]

dbesseli_dx(v, x) = autodiff(Forward, _x->besseli(v, _x), 
                             Duplicated, Duplicated(x, 1.0))[2]


for line in eachline("data/besseli/enzyme/besseli_enzyme_tests.csv")
    (v, x, dv, dx) = parse.(Float64, split(line))
    test_dv = dbesseli_dv(v, x)
    test_dx = dbesseli_dx(v, x)
    # TODO (cg 2023/05/30 12:09): temporarily test at lower rtols in the v=0.001
    # case. The power series code tests for convergence scaled by the series
    # value itself, which costs a little bit of rtol. When x is about 20, we
    # just barely hit that edge regime where we switch to the large argument
    # expansion and that edge zone also costs a digit. Those are things to
    # discuss addressing in a different PR I think.
    if v == 0.001 && x < 21.0
      @test isapprox(dv, test_dv, rtol=1e-11)
      @test isapprox(dx, test_dx, rtol=1e-11)
    else
      @test isapprox(dv, test_dv, rtol=5e-14)
      @test isapprox(dx, test_dx, rtol=5e-14)
    end
end
