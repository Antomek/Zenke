using Plots, Distributions, Zygote, OMEinsum, Optimization, OptimizationOptimisers, ChainRules, ChainRulesCore
using Flux: logitcrossentropy, logitbinarycrossentropy, onehotbatch

input_number = 100;
hidden_number = 4;
output_number = 2;

Δt = 1e-3
step_number = 200;

batch_size = 256 # batch size for training;

begin
	τ_syn = 5e-3 # s; synaptic time constant
	τ_mem = 10e-3 # s; membrane time constant
	τ_out = 20e-3 # s; output layer time constant
end;

β(τ) = exp(- Δt / τ)

begin
	β_syn = β(τ_syn)
	β_mem = β(τ_mem)
	β_out = β(τ_out)
end;

X_data = let
	freq = 5 # Hz
	prob = freq * Δt
	mask = rand(Uniform(0, 1), (batch_size, step_number, input_number))
	x_data = (mask .< prob)
end;

Y_data = (rand(batch_size) .< 0.5) .+ 1

input_spikeplot = let
	heatmap(transpose(X_data[1, : ,:]), legend=false, c = :grayC, 
		aspect_ratio = :equal)
	xlims!(0, step_number)
	ylims!(0, input_number)
	xlabel!("Time (ms)")
	ylabel!("Unit")
end

LIF_W1_init, LIF_W2_init = let
    weight_scale = Float32(7 * (1 - β_mem))
	
    W1 = rand(Normal(0, weight_scale / Float32(sqrt(input_number))), (input_number, hidden_number))
    W2 = rand(Normal(0, weight_scale / Float32(sqrt(hidden_number))), (hidden_number, output_number))
    (W1, W2)
end;

function layer_inputs(input, W)
    ein"abc,cd->abd"(input, W)
end

input_plot = let
	dims = (1, 1)
	p = []
	colors = palette(:Dark2_5)
	for i in 1:prod(dims)
        push!(p, plot(layer_inputs(X_data, LIF_W1_init)[:, i, :], palette = :Dark2_5, fmt=:svg, linewidth = 2))
	end
	plot(p..., layout = dims, grid = false, legend = false, size=(1000, 600))
end

Θ(x) = x > 0f0 ? 1f0 : 0f0

Θ_spike(x) =  x > 0f0 ? 1f0 : 0f0

function dSuperSpike(x; scale = 100f0)
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

function LIF_Zenke_simulation(X, W1, W2; surrogate = true)
    synapse_state = zeros((batch_size, hidden_number))
    membrane_state = zeros((batch_size, hidden_number))

    # Here we define two lists which we use to record the membrane potentials and output spikes
    membrane_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)
    spike_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)

    h1 = layer_inputs(X, W1)

    spike_fun = (surrogate ? Θ_spike : Θ)

    # Here we loop over time
    for t in 1:step_number
        membrane_threshold = membrane_state .- 1.0
        out = spike_fun.(membrane_threshold)

        membrane_buff[:, t, :] = membrane_state
        spike_buff[:, t, :] = out

        new_synapse_state = @. β_syn * synapse_state + h1[:, t, :]
        new_membrane_state = @. (β_mem * membrane_state + synapse_state) * (1.0 - out)

        synapse_state = new_synapse_state
        membrane_state = new_membrane_state
    end

    membrane_rec = copy(membrane_buff)
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

    return membrane_rec, spike_rec, network_outputs
end

membrane_record, spike_record, output_record = LIF_Zenke_simulation(X_data, LIF_W1_init, LIF_W2_init);

function plot_voltages(membrane_record; spikes = nothing, spike_height = 2., dim = (2, 2))
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

function LIF_classification_accuracy(X, Y, weights; surrogate = true)
    W1 = weights[1:100, :]
    W2 = transpose(weights[101:end, :])

    output = LIF_Zenke_simulation(X, W1, W2; surrogate = surrogate)[3]
    time_max = maximum(output, dims = 2) # Maximum over time
    y_network = dropdims(time_max; dims = 2)
    unit_max = getindex.(Tuple.(argmax(y_network; dims = 2)), 2)
	accuracy = mean(unit_max .== Y)
    return accuracy
end

function LIF_SNN_loss(weights; surrogate = true)
    W1 = weights[1:100, :]
    W2 = transpose(weights[101:end, :])

	output = LIF_Zenke_simulation(X_data, W1, W2; surrogate = surrogate)[3]
	time_max = maximum(output, dims = 2)
    y_network = transpose(dropdims(time_max; dims = 2))
    y_label = onehotbatch(Y_data, 1:2)
	return logitcrossentropy(y_network, y_label)
end

LIF_surrogate_loss_record = Float32[]
LIF_surrogate_weights = Matrix{Float32}[]
LIF_no_surrogate_loss_record = Float32[]
LIF_no_surrogate_weights = Matrix{Float32}[]

callback_maker = function(;surrogate = true)
    return callback = function (p, l)
    display(l)
    surrogate ? push!(LIF_surrogate_loss_record, l) : push!(LIF_no_surrogate_loss_record, l)
    surrogate ? push!(LIF_surrogate_weights, p) : push!(LIF_no_surrogate_weights, p)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
    end
end

adtype = Optimization.AutoZygote()
LIF_surrogate_optf = Optimization.OptimizationFunction((x,p) -> LIF_SNN_loss(x; surrogate = true), adtype)
LIF_surrogate_optprob = Optimization.OptimizationProblem(LIF_surrogate_optf, vcat(LIF_W1_init, transpose(LIF_W2_init)))

LIF_no_surrogate_optf = Optimization.OptimizationFunction((x,p) -> LIF_SNN_loss(x; surrogate = false), adtype)
LIF_no_surrogate_optprob = Optimization.OptimizationProblem(LIF_no_surrogate_optf, vcat(LIF_W1_init, transpose(LIF_W2_init)))

epochs = 1000

surrogate_result = Optimization.solve(LIF_surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = true), maxiters = epochs)
no_surrogate_result = Optimization.solve(LIF_no_surrogate_optprob, Optimisers.Adam(2f-3, (9f-1, 9.99f-1)), callback  = callback_maker(surrogate = false), maxiters = epochs)

loss_plot = plot(1:(epochs+1), LIF_surrogate_loss_record, xlabel = "Epochs", ylabel = "Crossentropy loss"; color = :orange, lw=3, label = "Surrogate", ylims = (0., 0.9), yticks = 0:0.1:0.9)
plot!(loss_plot, 1:(epochs+1), LIF_no_surrogate_loss_record, xlabel = "Epochs", ylabel = "Crossentropy loss"; color = :blue, lw=3, label = "No surrogate")

@info "Surrogate accuracy: " LIF_classification_accuracy(X_data, Y_data, LIF_surrogate_weights[end])
@info "No surrogate accuracy: " LIF_classification_accuracy(X_data, Y_data, LIF_no_surrogate_weights[end]; surrogate = false)

using CairoMakie, LaTeXStrings

SuperSpike(x, β = 100f0) = 1f0 / (β * abs(x)^2 + 1)

surrogate_gradient_plot = let
    fig = Figure(resolution = (1000, 600), fontsize = 20)

    colors = Makie.wong_colors()

    ax = Axis(fig[2,1:8], xlabel = "Epoch", ylabel = "Crossentropy loss")

    lines!(ax, 1:(epochs+1), LIF_surrogate_loss_record; color = colors[1], label = "Surrogate gradient \n training", linewidth = 3)
    lines!(ax, 1:(epochs+1), LIF_no_surrogate_loss_record; color = colors[2], label = "Actual gradient \n training", linewidth = 3)

    ax.yticks = 0:0.1:0.9
    ax.xticks = 0:100:epochs

    hidedecorations!(ax, ticks = false, ticklabels = false, label = false)
    hidespines!(ax, :r, :t)

    Legend(fig[2, 9:10], ax, framevisible = false)

    xs = -1:0.001:1

    activation_ax = Axis(fig[1,1:5], xlabel = "x")

    step_vals = [Θ_spike(x) for x in xs]
    surrogate_forward_vals = [SuperSpike(x) for x in xs]

    lines!(activation_ax, xs, step_vals; color = colors[1], label = L"H(x)", linewidth = 3)
    lines!(activation_ax, xs, surrogate_forward_vals; color = colors[2], label = L"\text{SuperSpike}(x)", linewidth = 3)

    grad_ax = Axis(fig[1,6:10], xlabel = "x")

    surrogate_grad_vals = [dSuperSpike(x) for x in xs]

    lines!(grad_ax, xs, surrogate_grad_vals; color = colors[2], label = L"\frac{d \, \text{SuperSpike}}{d x}", linewidth = 3)
    vlines!([0.]; color = :gray, linestyle = :dot, label = L"\frac{d H}{d x}", linewidth = 3)

    hidedecorations!.([grad_ax, activation_ax], ticks = false, ticklabels = false, label = false)
    hidespines!.([grad_ax, activation_ax], :r, :t)

    axislegend.([activation_ax, grad_ax], position = :rt, framevisible = false)

    fig
end
