using Plots, Distributions, Zygote, OMEinsum

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

W1_init, W2_init = let
    weight_scale = Float32(7 * (1 - β_mem))
	
    W1 = rand(Normal(0, weight_scale / Float32(sqrt(input_number))), (input_number, hidden_number))
    W2 = rand(Normal(0, weight_scale / Float32(sqrt(hidden_number))), (hidden_number, output_number))
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
        push!(p, plot(sum(layer_inputs(X_data, W1_init), dims = 2), palette = :Dark2_5, fmt=:svg, linewidth = 2))
	end
	plot(p..., layout = dim, grid = false, legend = false, size=(1000, 600))
end

function spike_fun(x)
    res = Zygote.Buffer(x, size(x)...)
    for i in eachindex(x)
        x[i] > 0. ? res[i] = 1. : res[i] = 0.
    end
    return copy(res)
end

function simulation(X, W1, W2)
    synapse_state = zeros((batch_size, hidden_number))
    membrane_state = zeros((batch_size, hidden_number))

    # Here we define two lists which we use to record the membrane potentials and output spikes
    membrane_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)
    spike_buff = Zygote.Buffer(W1, batch_size, step_number, hidden_number)

    h1 = layer_inputs(X, W1)

    # Here we loop over time
    for t in 1:step_number
        membrane_threshold = membrane_state .- 1.0
        out = spike_fun(membrane_threshold)
        reset = out #.detach() # We do not want to backprop through the reset

        new_synapse_state = @. β_syn * synapse_state + h1[:, t, :]
        new_membrane_state = @. (β_mem * membrane_state + synapse_state) * (1.0 - out)
        
        membrane_buff[:, t, :] = membrane_state
        spike_buff[:, t, :] = out
        
        membrane_state = new_membrane_state
        synapse_state = new_synapse_state
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

membrane_record, spike_record, output_record = simulation(X_data, W1_init, W2_init);

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
