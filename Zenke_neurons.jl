create_syn_vars(num_incoming) = [@variables $el(t) for el in [Symbol(:I_syn, i) for i = 1:num_incoming]]
var_sum(variables) = sum(reduce(vcat, variables))
syn_sum(num_incoming, syns) = num_incoming == 0 ? 0.0 : sum(reduce(vcat, syns))

function LIF_neuron(; name=name, num_incoming::Integer = 0, I_applied::Function = t -> 0.)
    internal_states = @variables Ṽ(t)=0. V(t)=0. I(t)=0. S(t)=0.
    applied_current = @variables I_app(t) = 0.0

    syns = create_syn_vars(num_incoming)

    params = @parameters τ_V = 6.0 θ = 0.99 mask = true

    eqs = [
        D(Ṽ) ~ (-V + I_app + syn_sum(num_incoming, syns)) / τ_V, 
        S ~ ϕ(V - θ),
        V ~ Ṽ - my_floor(Ṽ),
        I_app ~ mask * I_applied(t)
    ]

    syn_defaults = Dict(syn... => 0. for syn in syns)
    states = cat(internal_states, applied_current, syns..., dims = 1)

    return ODESystem(eqs, t, states, params; name=name)
end

function LI_neuron(; name=name, num_incoming::Integer = 0, I_applied::Function = t -> 0.)
    internal_states = @variables Ṽ(t)=0. V(t)=0. I(t)=0. S(t)=0.
    applied_current = @variables I_app(t) = 0.0

    syns = create_syn_vars(num_incoming)

    params = @parameters τ_V = 6.0 θ = 0.99 mask = true

    eqs = [
        D(Ṽ) ~ (-V + I_app + syn_sum(num_incoming, syns)) / τ_V, 
        S ~ ϕ(V - θ),
        V ~ Ṽ,
        I_app ~ mask * I_applied(t)
    ]

    syn_defaults = Dict(syn... => 0. for syn in syns)
    states = cat(internal_states, applied_current, syns..., dims = 1)

    return ODESystem(eqs, t, states, params; name=name)
end
