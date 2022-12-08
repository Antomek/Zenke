using OrdinaryDiffEq, ModelingToolkit, ChainRules, ChainRulesCore, Plots
gr()

function new_gradient(x)
    #println("My gradient.")
    return conj((abs(x / 2) + 1)^(-2))
end

function H(a)
    #return (sign(a) + 1) / 2
    return (a ≥ 0. ? 1. : 0.)
end

function ϕ(x)
   return H(x) 
end

@scalar_rule ϕ(x) new_gradient(x)

@register_symbolic ϕ(x)
@register_symbolic H(x)

f(y) = sqrt(π / 9.) * exp(- π^2 * y^2 / 9)
function pulse(x, c)
    ϵ = 0.1
    return 0.5 * (1 / ϵ) * f((x - c) / ϵ)
end

function δ_approx_1(x, c)
    b = 1e-2
    (1 / (b * √π)) * exp(- ((x - c) / b)^2)
end

function δ_approx_2(x, c)
    x == 1 ? 1. : 0.
end

#@register_symbolic δ_approx_2(x, c)

function my_floor(x)
    (x > 0) * floor(x)
end

function my_floor_test(Ṽ)
    (floor(Ṽ) > 0.5) * floor(Ṽ)
end

function ∇my_floor(x)
    return (x > 0) * new_gradient(rem(x,1) - 1)
end

@scalar_rule my_floor(x) ∇my_floor(x)

@variables t
D = Differential(t)

@register_symbolic my_floor(x)

function node_sys(;name)
    states = @variables V(t)=0. I(t)=0. S(t)=0. test(t) = 0.
    params = @parameters τ_V = 1/5. θ = 1.
    eqs = [
        D(V) ~ (-V + I) / τ_V, #- δ_approx_1(V, θ),
        S ~ δ_approx_1(V, θ),
        D(test) ~ S * D(V)
    ]

    #events = [Ṽ - floor(Ṽ) ~ θ]
    #affect = [V ~ 0.]

    return ODESystem(eqs, t, states, params; name=name)#, continuous_events = events)
end

function link(network, sys_pre, sys_post; name = ModelingToolkit.getname(network))
    states = @variables S(t)=0.0 I(t)=0.0
    params = @parameters w = 8. [tunable = true] τ_connection=0.5

    linkeqs = [
        D(I) ~ -I / τ_connection + w * S,
    ]
    #events = [test ~ 0]

    prename, postname = ModelingToolkit.getname.([sys_pre, sys_post])
    link_sys = ODESystem(linkeqs, t, states, params; name = Symbol(prename, :_to_, postname))

    oldeqs = ModelingToolkit.get_eqs(network)

    neweqs = [
        link_sys.S ~ sys_pre.S,
        link_sys.I ~ sys_post.I
    ]


    eqs = cat(oldeqs, neweqs, dims=1)
    systems = cat(ModelingToolkit.get_systems(network), link_sys, dims=1)

    return ODESystem(eqs, t, [], []; name = name, systems = systems)
end

input_node = let
    @named node = node_sys()
    @variables  I(t)
    eq = [I ~ pulse(t, 5.) - pulse(t, 15.)]
    extend(ODESystem(eq, t, [], []; name = :input_node), node)
end

@named output_node = node_sys()

network = let
    network = ODESystem(Array{Equation,1}(undef, 0), t, [], []; name=:network, systems = [input_node, output_node])
    network = link(network, input_node, output_node)
    network = structural_simplify(network)
end

prob = ODEProblem(network, [], (0.0, 30.0), [])
sol = solve(prob, Tsit5(); dtmax = 1e-2, reltol = 1e-7, abstol = 1e-7)

V_in_sym = input_node.V
#Ṽ_in_sym = input_node.Ṽ
test_in_sym = input_node.test
S_in_sym = input_node.S
I_out_sym = network.input_node_to_output_node₊I
#Ṽ_out_sym = output_node.Ṽ
V_out_sym = output_node.V
S_out_sym = output_node.S
test_out_sym = output_node.test

variable_plot = let
    p1 = plot(sol; idxs = V_in_sym, color = :lightgreen, label = "V_pre")
    p2 = plot(sol; idxs = S_sym, color = :purple, label = "S")
    p3 = plot(sol; idxs = test_in_sym, color = :red, label = "∫S_in dt")
    p4 = plot(sol; idxs = I_sym, color = :orange, label = "I")
    p5 = plot(sol; idxs = V_out_sym, color = :darkred, label = "V_out")
    p6 = plot(sol; idxs = S_out_sym, color = :pink, label = "S_out")
    p7 = plot(sol; idxs = test_out_sym, color = :blue, label = "∫S_out dt")

    Plots.plot(p1, p2, p3, p4, p5, p6, p7; layout = (7,1), size = (400., 900))
end

#function split_parameters(params)
#    indexmap = [p => i for (p,i) in zip(params, 1:length(params))]
#    weight_indexmap = eltype(indexmap)[]
#    for (p, i) in indexmap
#        istunable(p) && push!(weight_indexmap, p => i)
#    end
#    return indexmap, weight_indexmap
#end
#
#p = prob.p
#param_idxs, weight_idxs = split_parameters(parameters(network))
#indexof(sym,syms) = findfirst(isequal(sym),syms)
#output_idxs = indexof(Ṽ_out_sym, states(network))
#p_subset_indices = last.(weight_idxs)
#idx_to_subset_idx(i) = findfirst(isequal(i), p_subset_indices)
#
#using Flux, Zygote, SciMLSensitivity
#
#function remake_solution(p_subset)
#    new_p = [i in p_subset_indices ? p_subset[idx_to_subset_idx(i)] : p[i] for i in eachindex(p)]
#    _prob = remake(prob, p = new_p)
#    solve(_prob, Tsit5(); dtmax = 0.1, reltol = 1e-6, abstol = 1e-6, sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
#end
#
#function test_loss(param_subset)
#    _sol = remake_solution(param_subset)
#    sum(_sol[output_idxs, :])
#end
#function test_loss2(param_subset)
#    _sol = remake_solution(param_subset)
#    maximum(_sol[output_idxs, :])
#end
#
#p_subset = p[p_subset_indices]
#lossval = test_loss(p_subset)#p_subset)
#dp1 = Zygote.gradient(test_loss, p_subset)
#dp2 = Zygote.gradient(test_loss2, p_subset)
#
#training_result = let
#    new_w = copy(p_subset)
#    λ = 100.
#    for i in 1:20
#        new_w = new_w - λ * Zygote.gradient(test_loss2, new_w)[1]
#        @info "Loss: $(test_loss(new_w))"
#        @info " w: $(new_w)"
#    end
#    new_w
#end
#
#remade_solution = remake_solution(training_result)
#remade_variable_plot = let
#    p1 = plot(remade_solution; idxs = Ṽ_in_sym, color = :forestgreen, label = "Ṽ_pre", yrange = (-1., 5.))
#    p2 = plot(remade_solution; idxs = V_in_sym, color = :lightgreen, label = "V_pre", yrange = (0., 1.))
#    p3 = plot(remade_solution; idxs = S_sym, color = :purple, label = "S", yrange = (0., 1.))
#    p3 = plot(remade_solution; idxs = S_sym, color = :purple, label = "S", yrange = (0., 1.))
#    p4 = plot(remade_solution; idxs = I_sym, color = :orange, label = "I")
#    p5 = plot(remade_solution; idxs = Ṽ_out_sym, color = :red, label = "Ṽ_post", yrange = (-1., 5.))
#    p6 = plot(remade_solution; idxs = V_out_sym, color = :darkred, label = "V_out", yrange = (0., 1.))
#    plot(p1, p2, p3, p4, p5, p6; layout = (6,1), size = (400., 900))
#end
#
#output_plot = let
#    p1 = plot(remade_solution; idxs = Ṽ_out_sym, color = :orange, label = "Ṽ_post")
#    p2 = plot(remade_solution; idxs = V_out_sym, color = :blue, label = "V_out")
#    plot(p1, p2, layout = (2,1))
#end
