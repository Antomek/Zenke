Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}

function SuperSpike(x)
    x = x/2
    u = x*(abs(x) - 1)/(x^2-1)
    return (u+1)*0.5
end

# Superspike sigmoid derivative
function dSuperSpike(x)
    return conj((abs(x / 2) + 1)^(-2))
end

## The scalar case:

# Heaviside step function
function H(a)
    return (sign(a) + 1) / 2
end

# Smooth step
function forward_ϕ(x)
    y = 1e3 * x
    0.5 * (y / (1 + abs(y)) + 1)
end

# We'll call our surrogate nonlinearity "φ"
function ϕ(x)
   return H(x) 
end

# We need to define a new overload of ChainRulesCore.rrule to handle
# computation nodes that evaluate our surrogate binary transfer function
@scalar_rule(ϕ(x), dSuperSpike(x))
@register_symbolic ϕ(x)

function ChainRulesCore.rrule(::typeof(Broadcast.broadcasted),
                         ::typeof(ϕ), x::Numeric)

    Ω = ϕ.(x)

    function broadcasted_ϕ_pullback(dΩ)
        x_thunk = InplaceableThunk(
           dx -> @.(dx += dΩ * dSuperSpike(x)),
           @thunk @.(dΩ * dSuperSpike(x))
           )
        NoTangent(), NoTangent(), x_thunk
    end
    return Ω, broadcasted_ϕ_pullback
end

function my_floor(x)
    (x > 0 ) * floor(x)
end

function ∇my_floor(x)
    return (x > 0) * dSuperSpike(rem(x,1) - 1)
end

@scalar_rule my_floor(x) ∇my_floor(x)

@variables t
D = Differential(t)

@register_symbolic my_floor(x)
