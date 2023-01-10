using Plots, OrdinaryDiffEq, ModelingToolkit

Δt = 1e-3
t_end = 200.
step_number = Int(t_end ÷ Δt);


begin
    switch = 4

    C = 1 # ms
    τ_s = 10
    τ_u = 100
    V_0 = -40 # unitless
    V_s0 = [-41, -39, -38.5, -39, -38.5][switch]
    V_u0 = [-50, -50, -50, -54.5, -54.5][switch]
    g_f = 1
    g_s = 0.5
    g_u = 0.015
    V_r = -40
    V_sr = -35
    ΔV_u = 3
    I = 5

    V_max = -30.
end

β(τ) = exp( -Δt / τ)

begin
    β_V = β(C / (2 * g_f * V_0))
    β_V_s = β(τ_s)
    β_V_u = β(τ_u)
end

V_∞(Vₙ, V_sₙ, V_uₙ, I) = (g_f * (Vₙ^2 + V_0^2) - g_s * (V_sₙ - V_s0)^2 - g_u * (V_uₙ - V_u0)^2 + I) / (2*g_f*V_0)

V_step(Vₙ, V_sₙ, V_uₙ, I) = V_∞(Vₙ, V_sₙ, V_uₙ, I) + (Vₙ - V_∞(Vₙ, V_sₙ, V_uₙ, I)) * β_V
V_s_step(V_sₙ, Vₙ) = Vₙ + (V_sₙ - Vₙ) * β_V_s
V_u_step(V_uₙ, Vₙ) = Vₙ + (V_uₙ - Vₙ) * β_V_u

Θ(x) = x > 0f0 ? 1f0 : 0f0

function MQIF_simulation(;I = 5.)
    V_state = V_0
    V_s_state = V_s0
    V_u_state = V_u0

    V_list = zeros(step_number)
    V_s_list = zeros(step_number)
    V_u_list = zeros(step_number)

    for t in 1:step_number
        V_threshold = V_state - V_max
        out = Θ(V_threshold)

        new_V_state = V_step(V_state, V_s_state, V_u_state, I) + out * (-V_step(V_state, V_s_state, V_u_state, I) + V_r)
        new_V_s_state = V_s_step(V_s_state, V_state) + out * (-V_s_step(V_s_state, V_state) + V_sr)
        new_V_u_state = V_u_step(V_u_state, V_state) + out * (-V_s_step(V_u_state, V_state) + V_u_state + ΔV_u)

        V_list[t] = V_state
        V_s_list[t] = V_s_state
        V_u_list[t] = V_u_state

        V_state = new_V_state
        V_s_state = new_V_s_state
        V_u_state = new_V_u_state
    end

    return (V_list = V_list, V_s_list = V_s_list, V_u_list = V_u_list)
end

@variables t
D = Differential(t)

function MQIF_sys(;name)
    p1 = @parameters C=1. τ_s=10. τ_u=100. g_f=1. g_s=0.5 g_u=0.015
    p2 = @parameters V_0=-40. V_max=-30. V_s0=-38.5 V_u0=-50. V_r = -40. V_sr=-35. ΔV_u=3.
    params = vcat(p1, p2)

    vars = @variables V(t)=V_0 V_s(t)=V_s0 V_u(t)=V_u0

    eqs = [
    D(V) ~ (g_f * (V - V_0)^2 - g_s * (V_s - V_s0)^2 - g_u * (V_u - V_u0)^2 + I) / C,
    D(V_s) ~ (V - V_s) / τ_s,
    D(V_u) ~ (V - V_u) / τ_u
    ]

    root_eq = [V ~ V_max]
    affect_eq = [
    V ~ V_r,
    V_s ~ V_sr,
    V_u ~ V_u + ΔV_u
    ]

    return ODESystem(eqs, t, vars, params; name=name, continuous_events = root_eq => affect_eq)
end

@named MQIF_system = MQIF_sys()
MQIF_prob = ODEProblem(MQIF_system, [], (0.0, 200.0))
MQIF_sol = solve(MQIF_prob, Tsit5(); abstol = 1e-6, reltol = 1e-6)

records = MQIF_simulation(I = 5.)

mtk_Vplot = plot(MQIF_sol[MQIF_system.V], label = "V")
mtk_V_s_plot = plot(MQIF_sol[MQIF_system.V_s], label = "V_s")
mtk_V_u_plot = plot(MQIF_sol[MQIF_system.V_u], label = "V_u")
mtk_plot = plot(mtk_Vplot, mtk_V_s_plot, mtk_V_u_plot; layout = (3,1))

expEuler_V_plot = plot(1:step_number, records[:V_list], label = "V")
expEuler_V_s_plot = plot(1:step_number, records[:V_s_list], label = "V_s")
expEuler_V_u_plot = plot(1:step_number, records[:V_u_list], label = "V_u")
expEuler_plot = plot(expEuler_V_plot, expEuler_V_s_plot, expEuler_V_u_plot; layout = (3,1), xformatter = x -> string(floor(Int, x*Δt)))
