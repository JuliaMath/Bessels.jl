```julia
using ArbNumerics, DelimitedFiles, SpecialFunctions, Enzyme, Bessels

# Because you can't just use FiniteDifferences due to the "numerical noise".
simplefd(f,x,h=ArbReal(1e-40)) = (f(x+h)-f(x))/h

ArbNumerics.setprecision(ArbReal, digits=100)
arb_besseli(v,x)  = ArbNumerics.besseli(ArbReal(v), ArbReal(x))

# TODO (cg 2023/05/30 11:44): temporarily, we make vgrid[end]=15.1 instead of
# 15.0 or some other integer because the integer-order `besseli` derivatives
# need to be fixed with dispatch, like `besselk`.
vgrid = range(1e-3, 15.1,  length=20) 
xgrid = range(1e-3, 100.0, length=30)
vx = vec(collect(Iterators.product(vgrid, xgrid)))

our_dx(v,x) = autodiff(Forward, _x->Bessels.besseli(v, _x), Duplicated, Duplicated(x, 1.0))[2]
our_dv(v,x) = autodiff(Forward, _v->Bessels.besseli(_v, x), Duplicated, Duplicated(v, 1.0))[2]

if !isinteractive()
  ref_values = map(vx) do vxj
    (v,x) = vxj
    dx = simplefd(_x->arb_besseli(v, _x), x) 
    dv = simplefd(_v->arb_besseli(_v, x), v) 
    Float64.((dv, dx))
  end

  out_matrix = hcat(getindex.(vx, 1),          # test v argument
                    getindex.(vx, 2),          # test x argument
                    getindex.(ref_values, 1),  # test d/dv value
                    getindex.(ref_values, 2))  # test d/dx value

  writedlm("besseli_enzyme_tests.csv", out_matrix)
end
```
