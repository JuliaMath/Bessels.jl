module BesselsEnzymeCoreExt

  using Bessels, EnzymeCore
  using EnzymeCore.EnzymeRules
  using Bessels.Math

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

  # This is fixing a straight bug in Enzyme.
  function EnzymeRules.forward(func::Const{typeof(sinpi)}, 
                               ::Type{<:Duplicated}, 
                               x::Duplicated)
      (sp, cp) = sincospi(x.val)
      Duplicated(sp, pi*cp*x.dval)
  end

  function EnzymeRules.forward(func::Const{typeof(sinpi)}, 
                               ::Type{<:Const}, 
                               x::Const)
      sinpi(x.val)
  end

end
