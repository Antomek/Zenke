using ModelingToolkit, OrdinaryDiffEq, Plots, BlockArrays, ChainRules, ChainRulesCore, Distributions, Flux

include("Zenke_neurons.jl")
include("Zenke_synapses.jl")
include("Zenke_networks.jl")
include("surrogate_gradient.jl")
include("Zenke_utils.jl")

@variables t
D = Differential(t)

input_layer_size = 100
hidden_layer_size = 4
output_layer_size = 2

tspan = (0.0, 200.0)
spiketimes = spiketime_generation(input_layer_size, tspan)
y_labels = Flux.onehot(rand() < 0.5, 0:1)

inputspike_plot = let
    p = plot(xlabel = "Time (ms)", ylabel = "Spike #", legend = false)
    for i in eachindex(spiketimes)
        if !isempty(spiketimes[i])
            for tₛ in spiketimes[i]
                scatter!(p, (tₛ, i), color = :black, shape = :rect, markersize = 2.)
            end
        end
    end
    p
end

input_currents = [t -> spiketrain(tₛs)(t) for tₛs in spiketimes]

function nodes(i)
    (i ∈ 1:input_layer_size) && return n -> LIF_neuron(num_incoming = n, name = Symbol(:input_, i), I_applied = input_currents[i])
    (i ∈ 1:(input_layer_size .+ hidden_layer_size)) && return n -> LIF_neuron(num_incoming = n, name = Symbol(:hidden_, i))
    (i ∈ 1:(input_layer_size .+ hidden_layer_size .+ output_layer_size)) && return n -> LI_neuron(num_incoming = n, name = Symbol(:output_, i))
end

W1, W2 = weight_init(input_layer_size, hidden_layer_size, output_layer_size)

Adjacency_matrix = PseudoBlockArray{Tuple{Synapse}}(undef, [input_layer_size, hidden_layer_size, output_layer_size], [input_layer_size, hidden_layer_size, output_layer_size])
for block in blocks(Adjacency_matrix)
    block .= [(EmptyConnection(),)]
end
Adjacency_matrix[Block(1,2)] .= weights_to_syns(140 .* W1; tunable = true)
Adjacency_matrix[Block(2,3)] .= weights_to_syns(20 .* W2; tunable = true)

function edges(i, j)
    return Adjacency_matrix[i, j]
end

@time network = build_network(nodes, edges, 1:(input_layer_size + hidden_layer_size + output_layer_size); name = :network)
@time prob = ODEProblem(network, [], tspan, []; jac = true, sparse = true)
@time sol = solve(prob, Tsit5(); abstol = 1e-6, reltol = 1e-6, dtmax = 0.1)

input_voltagesyms = [sym for sym in filter_variables(all_states(network), ["input", "V"]) if !sym_contains(sym, "pre") && !sym_contains(sym, "post")]
hidden_voltagesyms = [sym for sym in filter_variables(all_states(network), ["hidden", "V"]) if !sym_contains(sym, "pre") && !sym_contains(sym, "post")]
I_to_first_hidden_unit = filter_variables(states(network), ["input", "hidden", "101"])
output_voltagesyms = [sym for sym in filter_variables(states(network), ["output"]) if !sym_contains(sym, "I_syn")]

input_Vplot = plot(sol, idxs = input_voltagesyms)
input_plot = plot(sol.t, sum(sol[I_to_first_hidden_unit]))
hidden_Vplot = plot(sol, idxs = hidden_voltagesyms)
output_Vplot = plot(sol, idxs = output_voltagesyms)

p = prob.p
param_idxs, weight_idxs = split_parameters(parameters(network))
indexof(sym,syms) = findfirst(isequal(sym),syms)
output_voltage_idxs = [indexof(sym, states(network)) for sym in output_voltagesyms]
p_subset_indices = last.(weight_idxs)
test_weights = [-5. for i in 1:length(p_subset_indices)]
idx_to_subset_idx(i) = findfirst(isequal(i), p_subset_indices)

using Zygote, SciMLSensitivity

function loss(p_subset)
    _sol = remake_solution(p_subset)
    max_output_Vs = maximum(_sol[output_voltage_idxs, :], dims = 2)
    Flux.logitcrossentropy(max_output_Vs, y_labels)
end

p_subset = p[p_subset_indices]
@info "Begin differentiating."
dp1 = Zygote.gradient(loss, p_subset)
