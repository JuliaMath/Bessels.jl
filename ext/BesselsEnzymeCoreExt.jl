module BesselsEnzymeCoreExt

  using Bessels, EnzymeCore
  using EnzymeCore.EnzymeRules
  using Bessels.Math
  import Bessels.Math: check_convergence

  # A manual method that separately transforms the `val` and `dval`, because
  # sometimes the `val` can converge while the `dval` hasn't, so just using an
  # early return or something can give incorrect derivatives in edge cases.
  function EnzymeRules.forward(config::FwdConfig,
                               func::Const{typeof(levin_transform)},
                               RT::Type{<:Union{Duplicated, DuplicatedNoNeed}},
                               s::Duplicated,
                               w::Duplicated)
      ls  = levin_transform(s.val,  w.val)
      dls = levin_transform(s.dval, w.dval)
      RT <: DuplicatedNoNeed ? dls : Duplicated(ls, dls)
  end

  # Series in `besseli`/`besselk` converge faster on the value than on the
  # dual; insist that *both* be converged before short-circuiting.
  function EnzymeRules.forward(config::FwdConfig,
                               func::Const{typeof(check_convergence)},
                               ::Type{<:Const},
                               t::Duplicated{T}) where{T}
    check_convergence(t.val) && check_convergence(t.dval)
  end

  function EnzymeRules.forward(config::FwdConfig,
                               func::Const{typeof(check_convergence)},
                               ::Type{<:Const},
                               t::Duplicated{T},
                               s::Duplicated{T}) where{T}
    check_convergence(t.val, s.val) && check_convergence(t.dval, s.dval)
  end

end
