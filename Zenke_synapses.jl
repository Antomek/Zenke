abstract type Synapse end

function get_name(ch::Synapse)
    Base.typename(ch |> typeof).name |> Symbol
end

struct EmptyConnection <: Synapse end

mutable struct link{T<:AbstractFloat} <: Synapse
    w::T
    S::T
    τ_synapse::T
    θ::T
    tunable::Bool
end

link(x; tunable::Bool=false) = link(x, 0., 12.5, 0.99, tunable)
syn_current(::link, sys::ODESystem) = sys.I_syn

function synapse(channel::link; name = name)
    states = @variables S(t) I_syn(t) V_pre(t) V_post(t)
    tunable = channel.tunable
    if tunable
        params = @parameters w [tunable = true, bounds = (0., Inf)]
    elseif !tunable
        params = @parameters w
    end

    eqs = [
        D(I_syn) ~ -I_syn / channel.τ_synapse + w * S,
        S ~ ϕ(V_pre - channel.θ)
        ]
    current = [eqs[2]]

    defaultmap = Dict(w => channel.w, I_syn => 0.)

    return ODESystem(eqs, t, states, params; defaults = defaultmap, observed = current, name = name)
end
