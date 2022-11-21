function remake_solution(p_subset)
    new_p = [i in p_subset_indices ? p_subset[idx_to_subset_idx(i)] : p[i] for i in eachindex(p)]
    _prob = remake(prob, p = new_p)
    solve(_prob, Tsit5(); dtmax = 0.1, reltol = 1e-6, abstol = 1e-6, sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
end

sym_contains(sym, string) = contains(String(Symbol(sym)), string)

function filter_variables(variables, list::Vector{String})
    [sym for sym in variables if all([sym_contains(sym, string) for string in list])]
end

function all_states(system::ODESystem)
    [map(eq -> eq.lhs, observed(system)); states(system)]
end

function split_parameters(params)
    indexmap = [p => i for (p,i) in zip(params, 1:length(params))]
    weight_indexmap = eltype(indexmap)[]
    for (p, i) in indexmap
        istunable(p) && push!(weight_indexmap, p => i)
    end
    return indexmap, weight_indexmap
end

function spiketime_generation(input_layer_size, timespan)
    μ = 1 / (5e-3) # avg ISI in ms

    ISIs = [Float32[] for i in 1:input_layer_size]
    spiketimes = [Float32[] for i in 1:input_layer_size]
    for i in 1:input_layer_size
        while (sum(ISIs[i]) + timespan[1]) < timespan[2]
            ISI = rand(Exponential(μ))
            (sum(ISIs[i]) + ISI + timespan[1]) > timespan[2] && break
            push!(ISIs[i], ISI)
        end
        spiketimes[i] = [sum(ISIs[i][1:j]) for j in eachindex(ISIs[i])]
    end
    
    return spiketimes
end

function data_generation(batch_size, input_layer_size, timespan)
    [(spiketime_generation(input_layer_size, timespan), rand(Bernoulli(0.5))) for i in 1:batch_size]
end

f_1(y) = abs(y) ≤ 1. ? (1 - abs(y)) : 0.
f_2(y) = sqrt(π / 9.) * exp(- π^2 * y^2 / 9)
function δ_approx(x, c)
    ϵ = 1
    return (1 / ϵ) * f_2((x - c) / ϵ)
end

function spiketrain(spiketimes)
    t -> sum([8. * δ_approx(t, tₛ) for tₛ in spiketimes])
end

function weight_init(input_layer_size, hidden_layer_size, output_layer_size)
    time_step = 1e-3
    τ_mem = 10e-3
    β = exp(-time_step / τ_mem)
    weight_scale = 7 * (1. - β)

    W1 = rand(Normal(0.0, weight_scale / sqrt(input_layer_size)), (input_layer_size, hidden_layer_size))
    W2 = rand(Normal(0.0, weight_scale / sqrt(hidden_layer_size)), (hidden_layer_size, output_layer_size))
    
    return W1, W2
end

function weights_to_syns(W::Matrix; tunable = false)
    array = Matrix{Tuple{Synapse}}(undef, size(W))
    for idx in eachindex(W)
        array[idx] = (link(W[idx]; tunable = tunable),)
    end
    return array
end

fullstates(sys) = [map(eq -> eq.lhs, observed(sys)); states(sys)]

"""
    create_network_graph(Ws::Array{<:AbstractMatrix})

Create a weighted, directed graph from an array of weights.
The weights must be matrices specifying the connection strenghts and size of the
layers.

We shall use the convention here that ``(W_{i})_{mxn}``, specifying the weights
from layer ``i`` to layer ``i+1``, means layer ``i`` has ``n`` neurons and layer ``i+1``
has ``m`` neurons (in a linear layer this implies you left multiply the weights
with the input, i.e. ``y = W*x``).

Returns `[g, edge_weights]`,
where `g` is the graph object and `edge_weights` the list of weights created
from the `Ws` argument.
"""
function create_network_graph(Ws::Array{<:AbstractMatrix})
    sizes = vcat([reverse(size(Ws[1]))...], [size(W, 1) for W in Ws[2:end]])

    startidxs = [sum(sizes[1:(i-1)]) + 1 for i in 1:length(sizes)]
    endidxs = [sum(sizes[1:i]) for i in 1:length(sizes)]
    vertexnums = [b:e for (b, e) in zip(startidxs, endidxs)]
    
    g = SimpleWeightedDiGraph(sum(sizes))

    for (i, (b, e, b_i, e_i)) in enumerate(zip(vertexnums[1:(end-1)], vertexnums[2:end], (startidxs .- 1)[1:(end-1)], endidxs[1:(end-1)]))
        for (s, s_i) in zip(b, b .- b_i)
            for (d, d_i) in zip(e, e .- e_i)
                !iszero(Ws[i][d_i, s_i]) ? add_edge!(g, s, d, Ws[i][d_i, s_i]) : nothing
            end
        end
    end

    edge_weights = getfield.(collect(edges(g)), :weight)
    g = SimpleDiGraph(g)

    return g, edge_weights
end

