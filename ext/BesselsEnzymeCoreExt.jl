module BesselsEnzymeCoreExt

  using Bessels, EnzymeCore
  using EnzymeCore.EnzymeRules
  using Bessels.Math
  import Bessels.Math: check_convergence

  # A manual method that separately transforms the `val` and `dval`, because
  # sometimes the `val` can converge while the `dval` hasn't, so just using an
  # early return or something can give incorrect derivatives in edge cases.
  function EnzymeRules.forward(func::Const{typeof(levin_transform)}, 
                               ::Type{<:Duplicated}, 
                               s::Duplicated,
                               w::Duplicated)
      (sv, dv, N) = (s.val, s.dval, length(s.val))
      ls  = levin_transform(sv, w.val)
      dls = levin_transform(dv, w.dval)
      Duplicated(ls, dls)
  end

  function EnzymeRules.forward(func::Const{typeof(check_convergence)},
                               ::Type{Const{Bool}},
                               t::Duplicated{T}) where{T}
    check_convergence(t.val) && check_convergence(t.dval)
  end

  function EnzymeRules.forward(func::Const{typeof(check_convergence)},
                               ::Type{Const{Bool}},
                               t::Duplicated{T},
                               s::Duplicated{T}) where{T}
    check_convergence(t.val, s.val) && check_convergence(t.dval, s.val)
  end

  # This will be fixed upstream: see #861 for Enzyme.jl whenever the next
  # release occurs.
  function EnzymeRules.forward(func::Const{typeof(sinpi)}, 
                               ::Type{<:Duplicated}, 
                               x::Duplicated)
      (sp, cp) = sincospi(x.val)
      Duplicated(sp, pi*cp*x.dval)
  end

  # #861 will probably also mean this can be deleted at the next release of
  # Enzyme.jl.
  function EnzymeRules.forward(func::Const{typeof(sinpi)}, 
                               ::Type{<:Const}, 
                               x::Const)
      sinpi(x.val)
  end



end
