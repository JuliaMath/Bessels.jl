
# This is just a placeholder at the moment to try and make CI trigger the error.
# This should hit the special method created in the EnzymeRules extension/weakdep.

using EnzymeCore, Enzyme
using Bessels.BesselFunctions
using BesselFunctions: besselkx_levin

dbesselk_dv(v, x) = autodiff(Forward, _v->besselkx_levin(_v, x, Val(20)), 
                             Duplicated, Duplicated(v, 1.0))

@assert isapprox(dbesselk_dv(1.5, 13.1), 0.0410929, atol=1e-5)

