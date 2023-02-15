using Plots, Distributions, Zygote, OMEinsum, Optimization, OptimizationOptimisers, ChainRules, ChainRulesCore
using Flux: logitcrossentropy, logitbinarycrossentropy, onehotbatch

Δt = 1e-3
#t_end = 20.
#step_number = Int(t_end ÷ Δt)
step_number = 200

input_number = 100;
hidden_number = 4;
output_number = 2;
batch_size = 2^7 # batch size for training;

begin
    switch = 1
    time_rescaling = 0.05

    C = 1 * time_rescaling # ms
    τ_s = 10 * time_rescaling
    τ_u = 100
    V_0 = -41 # unitless
    V_s0 = [-41, -39, -38.5, -39, -38.5][switch]
    V_u0 = [-50, -50, -50, -54.5, -54.5][switch]
    g_f = 1
    g_s = 0.5
    #g_u = 0.015
    g_u = 0.0
    V_r = -40
    V_sr = -35
    ΔV_u = 3
    I = 5

    V_max = -40.

	τ_syn = 5e-3 # s; synaptic time constant
	τ_mem = 10e-3 # s; membrane time constant
	τ_out = 20e-3 # s; output layer time constant
end

β(τ) = exp( -Δt / τ)

begin
    β_V = β(C / (2 * g_f * V_0))
    β_V_s = β(τ_s)
    β_V_u = β(τ_u)

	β_syn = β(τ_syn)
	β_mem = β(τ_mem)
	β_out = β(τ_out)
end

V_∞(Vₙ, V_sₙ, V_uₙ, I) = (g_f * (Vₙ^2 + V_0^2) - g_s * (V_sₙ - V_s0)^2 - g_u * (V_uₙ - V_u0)^2 + I) / (2*g_f*V_0)

V_step(Vₙ, V_sₙ, V_uₙ, I) = V_∞(Vₙ, V_sₙ, V_uₙ, I) + (Vₙ - V_∞(Vₙ, V_sₙ, V_uₙ, I)) * β_V
V_s_step(V_sₙ, Vₙ) = Vₙ + (V_sₙ - Vₙ) * β_V_s
V_u_step(V_uₙ, Vₙ) = Vₙ + (V_uₙ - Vₙ) * β_V_u

Θ(x) = x > 0f0 ? 1f0 : 0f0

X_data = let
	freq = 5 # Hz
	prob = freq * Δt
	mask = rand(Uniform(0, 1), (batch_size, step_number, input_number))
	x_data = (mask .< prob)
end;

Y_data = (rand(batch_size) .< 0.5) .+ 1

input_spikeplot = let
	heatmap(transpose(X_data[1, : ,:]), legend=false, c = :grayC, aspect_ratio = :equal)
	xlims!(0, step_number)
	ylims!(0, input_number)
	xlabel!("Time (ms)")
	ylabel!("Unit")
end

MQIF_W1_init, MQIF_W2_init = let
    input_weight_scale = 5. * Float32(((1 - β_V) / (2 * g_f * V_0))^(-1))
    output_weight_scale = Float32(7 * (1 - β_mem))
	
    W1 = rand(Normal(0, input_weight_scale / Float32(sqrt(input_number))), (input_number, hidden_number))
    W2 = rand(Normal(0, output_weight_scale / Float32(sqrt(hidden_number))), (hidden_number, output_number))
    (W1, W2)
end;

function layer_inputs(input, W)
    ein"abc,cd->abd"(input, W)
end

input_plot = let
	dim = (1, 1)
	p = []
	colors = palette(:Dark2_5)
	for i in 1:prod(dim)
        push!(p, plot(layer_inputs(X_data, MQIF_W1_init)[:, i, :], palette = :Dark2_5, fmt=:svg, linewidth = 2))
	end
	plot(p..., layout = dim, grid = false, legend = false, size=(1000, 600))
end

Θ(x) = x > 0f0 ? 1f0 : 0f0
Θ_spike(x) =  x > 0f0 ? 1f0 : 0f0

function dSuperSpike(x; scale = 110f0)
    return conj((scale * abs(x) + 1f0)^(-2))
end

Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}
function ChainRulesCore.rrule(::typeof(Broadcast.broadcasted),
                         ::typeof(Θ_spike), x::Numeric)

    Ω = Θ_spike.(x)

    function broadcasted_Θ_pullback(dΩ)
        x_thunk = InplaceableThunk(
           dx -> @.(dx += dΩ * dSuperSpike(x)),
           @thunk @.(dΩ * dSuperSpike(x))
           )
        NoTangent(), NoTangent(), x_thunk
    end
    return Ω, broadcasted_Θ_pullback
end

function MQIF_Zenke_simulation(X, W1, W2; surrogate = true)
    synapse_state = zeros((batch_size, hidden_number))
    V_hidden_state = fill(-44., (batch_size, hidden_number))
    V_s_hidden_state = fill(V_s0, (batch_size, hidden_number))

    # Here we define two lists which we use to record the membrane potentials and output spikes
    V_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)
    spike_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)

    h1 = layer_inputs(X, W1)

    spike_fun = (surrogate ? Θ_spike : Θ)

    # Here we loop over time
    for t in 1:step_number
        V_threshold = V_hidden_state .- V_max
        out = spike_fun.(V_threshold)

        if iseven(t)
            #new_V_hidden_state = @. V_step(V_hidden_state, V_s_hidden_state, V_u0, synapse_state) + out * (-V_step(V_hidden_state, V_s_hidden_state, V_u0, synapse_state) + V_r)
            new_V_hidden_state = @. V_step(V_hidden_state, V_s_hidden_state, V_u0, synapse_state) * (1 - out) + out * V_r
            new_hidden_synapse_state = @. β_syn * synapse_state + h1[:, t, :]

            V_hidden_state = new_V_hidden_state
            synapse_state = new_hidden_synapse_state
        elseif isodd(t)
            new_V_s_hidden_state = @. V_s_step(V_s_hidden_state, V_hidden_state) * (1 - out) + out * V_sr

            V_s_hidden_state = new_V_s_hidden_state
        end

        V_buff[:, t, :] = V_hidden_state
        spike_buff[:, t, :] = out

    end

    V_rec = copy(V_buff)
    spike_rec = copy(spike_buff)

    # Compute inputs to readout layer
	h2 = layer_inputs(spike_rec, W2)
	
	# To save the state of the output layer
	output_drive = zeros((batch_size, output_number))
	output_state = zeros((batch_size, output_number))
	
	# Array to record output state
	network_output_buff = Zygote.Buffer(W2, batch_size, step_number, output_number)
	
	# Loop over timesteps of output layer
	for t in 1:step_number
		new_output_drive = @. β_syn * output_drive + h2[:, t, :]
		new_output_state = @. β_mem * output_state + output_drive
		
		output_drive = new_output_drive
		output_state = new_output_state

		network_output_buff[:, t, :] = output_state
	end

    network_outputs = copy(network_output_buff)

    return V_rec, spike_rec, network_outputs
end

membrane_record, spike_record, output_record = MQIF_Zenke_simulation(X_data, MQIF_W1_init, MQIF_W2_init);

function plot_voltages(membrane_record; spikes = nothing, spike_height = -37., dim = (2, 2))
	data = copy(membrane_record)
	if spikes != nothing
		data[spikes .> 0.] .= spike_height
	end
	
	p = []
	colors = palette(:Dark2_5)
	for i in 1:prod(dim)
		push!(p, plot(data[i, :, :], palette = :Dark2_5, fmt=:svg, linewidth = 3))
	end
	plot(p..., layout = dim, grid = false, legend = false, size=(1000, 600))
end

hidden_layer_plots = plot_voltages(membrane_record; spikes = spike_record, dim = (2,2))
output_layer_plots = plot_voltages(output_record)

function MQIF_classification_accuracy(X, Y, weights; surrogate = true)
    W1 = weights[1:100, :]
    W2 = transpose(weights[101:end, :])

    output = MQIF_Zenke_simulation(X, W1, W2; surrogate = surrogate)[3]
    time_max = maximum(output, dims = 2) # Maximum over time
    y_network = dropdims(time_max; dims = 2)
    unit_max = getindex.(Tuple.(argmax(y_network; dims = 2)), 2)
	accuracy = mean(unit_max .== Y)
    return accuracy
end

function MQIF_SNN_loss(weights; surrogate = true)
    W1 = weights[1:100, :]
    W2 = transpose(weights[101:end, :])

	output = MQIF_Zenke_simulation(X_data, W1, W2; surrogate = surrogate)[3]
	time_max = maximum(output, dims = 2)
    y_network = transpose(dropdims(time_max; dims = 2))
    y_label = onehotbatch(Y_data, 1:2)
	return logitcrossentropy(y_network, y_label)
end
MQIF_no_surrogate_loss(weights) = MQIF_SNN_loss(weights; surrogate = false)

MQIF_surrogate_loss_record = Float32[]
MQIF_surrogate_weights = Matrix{Float32}[]
MQIF_no_surrogate_loss_record = Float32[]
MQIF_no_surrogate_weights = Matrix{Float32}[]

callback_maker = function(;surrogate = true)
    return callback = function (p, l)
    display(l)
    surrogate ? push!(MQIF_surrogate_loss_record, l) : push!(MQIF_no_surrogate_loss_record, l)
    surrogate ? push!(MQIF_surrogate_weights, p) : push!(MQIF_no_surrogate_weights, p)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
    end
end

adtype = Optimization.AutoZygote()
MQIF_surrogate_optf = Optimization.OptimizationFunction((x,p) -> MQIF_SNN_loss(x; surrogate = true), adtype)
MQIF_surrogate_optprob = Optimization.OptimizationProblem(MQIF_surrogate_optf, vcat(MQIF_W1_init, transpose(MQIF_W2_init)))

MQIF_no_surrogate_optf = Optimization.OptimizationFunction((x,p) -> MQIF_SNN_loss(x; surrogate = false), adtype)
MQIF_no_surrogate_optprob = Optimization.OptimizationProblem(MQIF_no_surrogate_optf, vcat(MQIF_W1_init, transpose(MQIF_W2_init)))

epochs = 200

#surrogate_result = Optimization.solve(surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = true), maxiters = epochs)
#no_surrogate_result = Optimization.solve(no_surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = false), maxiters = epochs)
MQIF_surrogate_result = Optimization.solve(MQIF_surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = true), maxiters = epochs)
MQIF_no_surrogate_result = Optimization.solve(MQIF_no_surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = false), maxiters = epochs)

MQIF_loss_plot = plot(1:(epochs+1), MQIF_surrogate_loss_record, xlabel = "Epochs", ylabel = "Crossentropy loss"; color = :orange, lw=3, label = "Surrogate", ylims = (0., 0.9), yticks = 0:0.1:0.9)
plot!(MQIF_loss_plot, 1:(epochs+1), MQIF_no_surrogate_loss_record, xlabel = "Epochs", ylabel = "Crossentropy loss"; color = :blue, lw=3, label = "No surrogate")

@info "Surrogate accuracy: " MQIF_classification_accuracy(X_data, Y_data, MQIF_surrogate_weights[end])
@info "No surrogate accuracy: " MQIF_classification_accuracy(X_data, Y_data, MQIF_no_surrogate_weights[end]; surrogate = false)
