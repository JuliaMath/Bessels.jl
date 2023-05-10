module BesselsEnzymeCoreExt

  # TODO (cg 2023/05/08 10:02): Compat of any kind. 

  using Bessels, EnzymeCore
  using EnzymeCore.EnzymeRules
  using Bessels.Math

  # A manual method that separately transforms the `val` and `dval`, because
  # sometimes the `val` can converge while the `dval` hasn't, so just using an
  # early return or something can give incorrect derivatives in edge cases.
  #
  # https://github.com/JuliaMath/Bessels.jl/issues/96 
  #
  # and links with for discussion.
  #
  # TODO (cg 2023/05/08 10:00): I'm not entirely sure how best to "generalize"
  # this to cases like a return type of DuplicatedNoNeed, or something being a
  # `Enzyme.Const`. These shouldn't in principle affect the "point" of this
  # function (which is just to check for convergence before applying a
  # function), but on its face this approach would mean I need a lot of
  # hand-written extra methods. I have an open issue on the Enzyme.jl repo at
  #
  # https://github.com/EnzymeAD/Enzyme.jl/issues/786
  #
  # that gets at this problem a bit. But it's a weird request and I'm sure Billy
  # has a lot of asks on his time.
  function EnzymeRules.forward(func::Const{typeof(levin_transform)}, 
                               ::Type{<:Duplicated}, 
                               s::Duplicated,
                               w::Duplicated)
      (sv, dv, N) = (s.val, s.dval, length(s.val))
      ls  = levin_transform(sv, w.val)
      dls = levin_transform(dv, w.dval)
      Duplicated(ls, dls)
  end

end
