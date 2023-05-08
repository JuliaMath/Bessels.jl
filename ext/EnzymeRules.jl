# TODO (cg 2023/05/08 10:38): As of now, this throws the error
#
# ERROR; LoadError: UndefVarError: `forward` not defined
#
# which is weird, because this code works fine in its own scope and not as an
# extension.

module EnzymeRules

  # TODO (cg 2023/05/08 10:02): Compat of any kind. 
  #
  # TODO (cg 2023/05/08 10:27): This only works on the master branch of Enzyme.
  # Which you can only use by doing
  # ] add EnzymeCore#main
  # ] add Enzyme#main
  # so that's not great.

  using Bessels, EnzymeCore
  using EnzymeCore.EnzymeRules
  using Bessels.Math

  # A manual method that takes an NTuple of partial sum terms and checks if it is
  # exactly converged before computing the Levin sequence transformation. If it is
  # exactly converged, the sequence transformation will create a divide-by-zero
  # problem. See
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
    ls  = (sv[N-1] == sv[N]) ? sv[N] : levin_transform(sv, w.val)
    dls = (dv[N-1] == dv[N]) ? dv[N] : levin_transform(dv, w.dval)
    Duplicated(ls, dls)
  end

end
