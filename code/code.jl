# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load packages
using LinearAlgebra: UniformScaling, I, diag, diagind, mul!, ldiv!, lu, lu!, norm
using SparseArrays: sparse, issparse, dropzeros!
using Printf: @sprintf

using SummationByPartsOperators

using LaTeXStrings
using CairoMakie
set_theme!(theme_latexfonts();
           fontsize = 26,
           linewidth = 3,
           markersize = 16,
           Lines = (cycle = Cycle([:color, :linestyle], covary = true),),
           Scatter = (cycle = Cycle([:color, :marker], covary = true),))

using PrettyTables: PrettyTables, pretty_table, ft_printf


const FIGDIR = joinpath(dirname(@__DIR__), "figures")
if !isdir(FIGDIR)
    mkdir(FIGDIR)
end


#####################################################################
# Utility functions

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) /
                      log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end


#####################################################################
# High-level interface of the equations and IMEX ode solver

rhs_stiff!(du, u, parameters, t) = rhs_stiff!(du, u, parameters.equation, parameters, t)
rhs_nonstiff!(du, u, parameters, t) = rhs_nonstiff!(du, u, parameters.equation, parameters, t)
operator(rhs_stiff!, parameters) = operator(rhs_stiff!, parameters.equation, parameters)
dot_entropy(u, v, parameters) = dot_entropy(u, v, parameters.equation, parameters)


# IMEX Coefficients
"""
    ARS111(T = Float64)

First-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS111{T} end
ARS111(T = Float64) = ARS111{T}()
function coefficients(::ARS111{T}) where T
    l = one(T)

    A_stiff = [0 0;
               0 l]
    b_stiff = [0, l]
    c_stiff = [0, l]
    A_nonstiff = [0 0;
                  l 0]
    b_nonstiff = [l, 0]
    c_nonstiff = [0, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


"""
    ARS222(T = Float64)

Second-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS222{T} end
ARS222(T = Float64) = ARS222{T}()
function coefficients(::ARS222{T}) where T
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)
    δ = 1 - 1 / (2 * γ)

    A_stiff = [0 0 0;
               0 γ 0;
               0 1-γ γ]
    b_stiff = [0, 1-γ, γ]
    c_stiff = [0, γ, 1]
    A_nonstiff = [0 0 0;
                  γ 0 0;
                  δ 1-δ 0]
    b_nonstiff = [δ, 1-δ, 0]
    c_nonstiff = [0, γ, 1]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443(T = Float64)

Third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443{T} end
ARS443(T = Float64) = ARS443{T}()
function coefficients(::ARS443{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               0 l/2 0 0 0;
               0 l/6 l/2 0 0;
               0 -l/2 l/2 l/2 0;
               0 3*l/2 -3*l/2 l/2 l/2]
    b_stiff = [0, 3*l/2, -3*l/2, l/2, l/2]
    c_stiff = [0, l/2, 2*l/3, l/2, l]
    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

# IMEX ARK solver
# This assumes that the stiff part is linear and that the stiff solver is
# diagonally implicit.
function solve_imex(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                    q0, tspan, parameters, alg;
                    dt,
                    relaxation = false,
                    callback = Returns(nothing),
                    save_everystep = false)
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    if save_everystep
        sol_q = [copy(q0)]
        sol_t = [first(tspan)]
    end
    y = similar(q) # stage value
    z = similar(q) # stage update value
    t = first(tspan)
    tmp = similar(q)
    k_stiff_q = similar(q) # derivative of the previous state
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        factor = a * dt
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        factorizations = Dict(factor => copy(factorization))
        W, factorization, factorizations
    end

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        # There are two possible formulations of a diagonally implicit RK method.
        # The "simple" one is
        #   y_i = q + h \sum_{j=1}^{i} a_{ij} f(y_j)
        # However, it can be better to use the smaller values
        #   z_i = (y_i - q) / h
        # so that the stage equations become
        #   q + h z_i = q + h \sum_{j=1}^{i} a_{ij} f(q + h z_j)
        # ⟺
        #   z_i - h a_{ii} f(q + h z_i) = \sum_{j=1}^{i-1} a_{ij} f(q + h z_j)
        # For a linear problem f(q) = T q, this becomes
        #   (I - h a_{ii} T z_i = a_{ii} T q + \sum_{j=1}^{i-1} a_{ij} T(q + h z_j)
        # We use this formulation and also avoid evaluating the stiff operator at
        # the numerical solutions (due to the large Lipschitz constant), but instead
        # rearrange the equation to obtain the required stiff RHS values as
        #   T(q + h z_i) = a_{ii}^{-1} (z_i - \sum_{j=1}^{i-1} a_{ij} f(q + h z_j))
        rhs_stiff!(k_stiff_q, q, parameters, t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i - 1)
                @. tmp += A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            # The right-hand side of the linear system formulated using the stages y_i
            # instead of the stage updates z_i would be
            #   @. tmp = q + dt * tmp
            # By using the stage updates z_i, we avoid the possibly different scales
            # for small dt.
            @. tmp = A_stiff[i, i] * k_stiff_q + tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(z, tmp)
            else
                factor = A_stiff[i, i] * dt

                F = let W = W, factor = factor,
                        factorization = factorization,
                        rhs_stiff_operator = rhs_stiff_operator
                    get!(factorizations, factor) do
                        fill!(W, 0)
                        W[diagind(W)] .= 1
                        @. W -= factor * rhs_stiff_operator
                        if issparse(W)
                            lu!(factorization, W)
                        else
                            factorization = lu!(W)
                        end
                        copy(factorization)
                    end
                end
                ldiv!(z, F, tmp)
            end

            # Compute new stage derivatives
            @. y = q + dt * z
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            else
                # The code below is equaivalent to
                #   rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
                # but avoids evaluating the stiff operator at the numerical solution.
                @. tmp = z
                for j in 1:(i-1)
                    @. tmp = tmp - A_stiff[i, j] * k_stiff[j] - A_nonstiff[i, j] * k_nonstiff[j]
                end
                @. k_stiff[i] = tmp / A_stiff[i, i]
            end
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        if relaxation
            @. y = dt * tmp # = qnew - q
            gamma = -2 * dot_entropy(q, y, parameters) / dot_entropy(y, y, parameters)
            @. q = q + gamma * y
            t += gamma * dt
        else
            @. q = q + dt * tmp
            t += dt
        end
        if save_everystep
            push!(sol_q, copy(q))
            append!(sol_t, t)
        end
        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    if save_everystep
        return (; u = sol_q,
                  t = sol_t)
    else
        return (; u = (q0, q),
                  t = (first(tspan), t))
    end
end

# this is specialized so that also the stage values of the last step are returned
function solve_imex_return_stages(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                                  q0, tspan, parameters, alg;
                                  dt,
                                  relaxation = false,
                                  callback = Returns(nothing),
                                  save_everystep = false)
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    if save_everystep
        sol_q = [copy(q0)]
        sol_t = [first(tspan)]
    end
    y = similar(q) # stage value
    z = similar(q) # stage update value
    stage_updates = Vector{typeof(q)}(undef, s) # y^i - u^n
    t = first(tspan)
    tmp = similar(q)
    k_stiff_q = similar(q) # derivative of the previous state
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        factor = a * dt
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        factorizations = Dict(factor => copy(factorization))
        W, factorization, factorizations
    end

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        if t + dt >= last(tspan)
            last_step = true
        else
            last_step = false
        end

        # There are two possible formulations of a diagonally implicit RK method.
        # The "simple" one is
        #   y_i = q + h \sum_{j=1}^{i} a_{ij} f(y_j)
        # However, it can be better to use the smaller values
        #   z_i = (y_i - q) / h
        # so that the stage equations become
        #   q + h z_i = q + h \sum_{j=1}^{i} a_{ij} f(q + h z_j)
        # ⟺
        #   z_i - h a_{ii} f(q + h z_i) = \sum_{j=1}^{i-1} a_{ij} f(q + h z_j)
        # For a linear problem f(q) = T q, this becomes
        #   (I - h a_{ii} T z_i = a_{ii} T q + \sum_{j=1}^{i-1} a_{ij} T(q + h z_j)
        # We use this formulation and also avoid evaluating the stiff operator at
        # the numerical solutions (due to the large Lipschitz constant), but instead
        # rearrange the equation to obtain the required stiff RHS values as
        #   T(q + h z_i) = a_{ii}^{-1} (z_i - \sum_{j=1}^{i-1} a_{ij} f(q + h z_j))
        rhs_stiff!(k_stiff_q, q, parameters, t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i - 1)
                @. tmp = tmp + A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            # The right-hand side of the linear system formulated using the stages y_i
            # instead of the stage updates z_i would be
            #   @. tmp = q + dt * tmp
            # By using the stage updates z_i, we avoid the possibly different scales
            # for small dt.
            @. tmp = A_stiff[i, i] * k_stiff_q + tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(z, tmp)
            else
                factor = A_stiff[i, i] * dt

                F = let W = W, factor = factor,
                        factorization = factorization,
                        rhs_stiff_operator = rhs_stiff_operator
                    get!(factorizations, factor) do
                        fill!(W, 0)
                        W[diagind(W)] .= 1
                        @. W -= factor * rhs_stiff_operator
                        if issparse(W)
                            lu!(factorization, W)
                        else
                            factorization = lu!(W)
                        end
                        copy(factorization)
                    end
                end
                ldiv!(z, F, tmp)
            end

            # Compute new stage derivatives
            @. y = q + dt * z
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            else
                # The code below is equaivalent to
                #   rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
                # but avoids evaluating the stiff operator at the numerical solution.
                @. tmp = z
                for j in 1:(i-1)
                    @. tmp = tmp - A_stiff[i, j] * k_stiff[j] - A_nonstiff[i, j] * k_nonstiff[j]
                end
                @. k_stiff[i] = tmp / A_stiff[i, i]
            end

            # Store stage value in the last step
            if last_step
                stage_updates[i] = copy(z)
            end
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        if relaxation
            @. y = dt * tmp # = qnew - q
            gamma = -2 * dot_entropy(q, y, parameters) / dot_entropy(y, y, parameters)
            @. q = q + gamma * y
            t += gamma * dt
        else
            @. q = q + dt * tmp
            t += dt
        end
        if save_everystep
            push!(sol_q, copy(q))
            append!(sol_t, t)
        end
        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    if save_everystep
        return (; u = sol_q,
                  t = sol_t,
                  stage_updates = stage_updates)
    else
        return (; u = (q0, q),
                  t = (first(tspan), t),
                  stage_updates = stage_updates)
    end
end



#####################################################################
# General interface

abstract type AbstractEquation end
Base.Broadcast.broadcastable(equation::AbstractEquation) = (equation,)


#####################################################################
# BBM discretization

struct BBM <: AbstractEquation end

get_u(u, equations::BBM) = u

function rhs_stiff!(du, u, equation::BBM, parameters, t)
    fill!(du, zero(eltype(du)))
    return nothing
end

operator(::typeof(rhs_stiff!), equation::BBM, parameters) = 0 * I

function rhs_nonstiff!(du, u, equation::BBM, parameters, t)
    (; D1, invImD2) = parameters
    tmp1 = parameters.tmp1
    tmp2 = parameters.tmp2
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear and quadratic invariants
    @. tmp1 = -one_third * u^2
    mul!(tmp2, D1, tmp1)
    mul!(tmp1, D1, u)

    # There are two normalizations of the BBM equation:
    # 1. u_t - u_{txx} + u_x + u u_x = 0
    # @. tmp2 += -one_third * u * tmp1 - tmp1
    # 2. u_t - u_{txx} + u u_x = 0
    @. tmp2 += -one_third * u * tmp1

    ldiv!(du, invImD2, tmp2)

    return nothing
end

function dot_entropy(u, v, equation::BBM, parameters)
    (; D1, D2, tmp1) = parameters
    mul!(tmp1, D2, v)
    half = one(eltype(u)) / 2
    @. tmp1 = half * u * (v - tmp1)
    return integrate(tmp1, D1)
end

function setup(q_func, equation::BBM, tspan, D, D2 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D2)
        D1 = D.central
        D2 = sparse(D.plus) * sparse(D.minus)
        invImD2 = lu(I - D2)
    elseif D isa FourierDerivativeOperator && isnothing(D2)
        D1 = D
        D2 = D1^2
        invImD2 = I - D2
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    q0 = q_func(tspan[1], x, equation)
    tmp1 = similar(q0)
    tmp2 = similar(q0)
    parameters = (; equation, D1, D2, invImD2, tmp1, tmp2)
    return (; q0, parameters)
end


#####################################################################
# BBMH discretization

struct BBMH{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::BBMH) = get_qi(q, equation, 0)
function get_qi(q, equation::BBMH, i)
    N = length(q) ÷ 3
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::BBMH, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))
    v = view(q, (1*N+1):(2*N))
    w = view(q, (2*N+1):(3*N))

    τ = equation.τ

    if D1 isa PeriodicUpwindOperators
        mul!(du, D1.central, v, -1)

        mul!(dv, D1.central, u)
        @. dv = (-dv + w) / τ

        @. dw = -v
    else
        mul!(du, D1, v)
        @. du *= -1

        mul!(dv, D1, u)
        @. dv = (-dv + w) / τ

        @. dw = -v
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::BBMH, parameters)
    D1 = parameters.D1
    τ = equation.τ

    if D1 isa PeriodicUpwindOperators
        D = sparse(D1.central)
        O = zero(D)
        jac = [O -D O;
               -1/τ*D O 1/τ*I;
               O -I O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O -D O;
               -1/τ*D O 1/τ*I;
               O -I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::BBMH, parameters, t)
    (; D1, tmp1) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))
    v = view(q, (1*N+1):(2*N))
    w = view(q, (2*N+1):(3*N))

    τ = equation.τ
    if D1 isa PeriodicUpwindOperators
        @. tmp1 = -one_third * u^2
        mul!(du, D1.central, tmp1)
        mul!(tmp1, D1.central, u)
        @. du += -one_third * u * tmp1

        @. dv = 0

        mul!(dw, D1.central, w, -τ)
    else
        @. tmp1 = -one_third * u^2
        mul!(du, D1, tmp1)
        mul!(tmp1, D1, u)
        @. du += -one_third * u * tmp1

        @. dv = 0

        mul!(dw, D1, w)
        @. dw *= -τ
    end

    return nothing
end

function dot_entropy(q1, q2, equation::BBMH, parameters)
    (; D1, tmp1) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0*N+1):(1*N))
    v1 = view(q1, (1*N+1):(2*N))
    w1 = view(q1, (2*N+1):(3*N))

    u2 = view(q2, (0*N+1):(1*N))
    v2 = view(q2, (1*N+1):(2*N))
    w2 = view(q2, (2*N+1):(3*N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp1 = half * (u1 * u2 + τ * v1 * v2 + w1 * w2)

    return integrate(tmp1, D1)
end

function setup(u_func, equation::BBMH, tspan, D1, D2 = nothing)
    if !isnothing(D2)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, BBM())
    v0 = zero(u0)

    # Use the discrete derivative operator to compute the initial condition
    # for w, in particular
    if D1 isa PeriodicUpwindOperators
        w0 = D1.central * u0
    else
        w0 = D1.central * u0
    end

    q0 = vcat(u0, v0, w0)

    tmp1 = similar(u0)

    parameters = (; equation, D1, tmp1)
    return (; q0, parameters)
end


#####################################################################
# BBMH -> BBM convergence test
gaussian(t, x::Number, equation::BBM) = 2 * exp(-0.02 * x^2)
gaussian(t, x::AbstractVector, equation::BBM) = gaussian.(t, x, equation)

function benjamin_bona_mahony_convergence(; latex = false,
                                            tspan = (0.0, 100.0),
                                            N = 2^10,
                                            accuracy_order = 7,
                                            alg = ARS443(),
                                            dt = 0.1 + 1.0e-14, # otherwise, the last step is so small that numerical errors pollute the convergence of q2
                                            kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -50.0
    xmax = 150.0

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_v_limit = Float64[]
    errors_w_limit = Float64[]

    u_limit, u_ini, u_limit_stage_updates = let equation = BBM()
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex_return_stages(rhs_stiff!, operator(rhs_stiff!, parameters),
                                             rhs_nonstiff!,
                                             q0, tspan, parameters, alg;
                                             dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation), sol.stage_updates
    end

    for τ in τs
        equation = BBMH(τ)
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex_return_stages(rhs_stiff!, operator(rhs_stiff!, parameters),
                                             rhs_nonstiff!,
                                             q0, tspan, parameters, alg;
                                             dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        # Specialized to a type 2 ARS method
        v_limit = zero(u_limit)
        a = let
            A_stiff, _ = coefficients(alg)
            Ainv = inv(A_stiff[2:end, 2:end])
            Ainv[end, :]
        end
        for i in eachindex(a)
            v_limit .-= a[i] .* (parameters.D1.central * u_limit_stage_updates[i + 1])
        end
        v = get_qi(sol.u[end], equation, 1)
        error_v = integrate(abs2, v - v_limit, parameters.D1) |> sqrt
        push!(errors_v_limit, error_v)

        w_limit = parameters.D1.central * u_limit
        w = get_qi(sol.u[end], equation, 2)
        error_w = integrate(abs2, w - w_limit, parameters.D1) |> sqrt
        push!(errors_w_limit, error_w)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical BBM solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"BBM solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_w_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_v_limit; label = L"Error $|\!|q_2(T\,) + \partial_{tx} u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv; position = :lt, framevisible = false)

    filename = joinpath(FIGDIR, "benjamin_bona_mahony_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# KdV discretization

struct KdV <: AbstractEquation end

get_u(u, equations::KdV) = u

function rhs_stiff!(du, u, equation::KdV, parameters, t)
    (; D3) = parameters

    mul!(du, D3, u)
    @. du = -du

    return nothing
end

operator(::typeof(rhs_stiff!), equation::KdV, parameters) = parameters.minus_D3

function rhs_nonstiff!(du, u, equation::KdV, parameters, t)
    (; D1, tmp) = parameters
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear and quadratic invariants
    @. tmp = -one_third * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du - one_third * u * tmp

    return nothing
end

function dot_entropy(u, v, equation::KdV, parameters)
    (; D1, tmp) = parameters
    @. tmp = u * v
    return 0.5 * integrate(tmp, D1)
end

function setup(u_func, equation::KdV, tspan, D, D3 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        minus_D3 = -D3
    elseif D isa PeriodicDerivativeOperator && D3 isa PeriodicDerivativeOperator
        D1 = D
        D3 = D3
        minus_D3 = -sparse(D3)
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    q0 = u0
    parameters = (; equation, D1, D3, minus_D3, tmp)
    return (; q0, parameters)
end


#####################################################################
# KdVH discretization

struct KdVH{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::KdVH) = get_qi(q, equation, 0)
function get_qi(q, equation::KdVH, i)
    N = length(q) ÷ 3
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::KdVH, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0 * N + 1):(1 * N))
    dv = view(dq, (1 * N + 1):(2 * N))
    dw = view(dq, (2 * N + 1):(3 * N))

    u = view(q, (0 * N + 1):(1 * N))
    v = view(q, (1 * N + 1):(2 * N))
    w = view(q, (2 * N + 1):(3 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        # du .= -D₊ * w
        mul!(du, D1.plus, w, -1)

        # dv .= (D * v - w) / τ
        mul!(dv, D1.central, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1.minus, u)
        @. dw = inv_τ * (-dw + v)
    else
        # du .= -D₊ * w
        mul!(du, D1, w)
        @. du = -du

        # dv .= (D * v - w) / τ
        mul!(dv, D1, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1, u)
        @. dw = inv_τ * (-dw + v)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KdVH, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(D)
        jac = [O O -Dp;
               O inv_τ*D -inv_τ*I;
               -inv_τ*Dm inv_τ*I O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*D -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::KdVH, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * u^2
    mul!(du, D, tmp)
    mul!(tmp, D, u)
    @. du = du - one_third * u * tmp

    fill!(dv, zero(eltype(dv)))
    fill!(dw, zero(eltype(dw)))

    return nothing
end

function dot_entropy(q1, q2, equation::KdVH, parameters)
    (; D1, tmp) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0 * N + 1):(1 * N))
    v1 = view(q1, (1 * N + 1):(2 * N))
    w1 = view(q1, (2 * N + 1):(3 * N))

    u2 = view(q2, (0 * N + 1):(1 * N))
    v2 = view(q2, (1 * N + 1):(2 * N))
    w2 = view(q2, (2 * N + 1):(3 * N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp = half * (u1 * u2 + τ * v1 * v2 + τ * w1 * w2)

    return integrate(tmp, D1)
end


function setup(u_func, equation::KdVH, tspan, D1, D3 = nothing)
    if !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, KdV())

    if D1 isa PeriodicUpwindOperators
        v0 = D1.minus * u0
        w0 = D1.central * v0
    else
        v0 = D1 * u0
        w0 = D1 * v0
    end

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# KdV KdVH convergence test
gaussian(t, x::Number, equation::KdV) = 2 * exp(-0.02 * x^2)
gaussian(t, x::AbstractVector, equation::KdV) = gaussian.(t, x, equation)

function korteweg_de_vries_convergence(; latex = false,
                                         tspan = (0.0, 100.0),
                                         N = 2^10,
                                         accuracy_order = 7,
                                         alg = ARS443(),
                                         dt = 0.05,
                                         kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -50.0
    xmax = 150.0

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]

    u_limit, u_ini = let equation = KdV()
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = KdVH(τ)
        (; q0, parameters) = setup(gaussian,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.central * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical KdV solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"KdV solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv; position = :lt, framevisible = false)

    filename = joinpath(FIGDIR, "korteweg_de_vries_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# KdV-Burgers discretization

struct KdVBurgers{T} <: AbstractEquation
    μ::T
end

get_u(u, equations::KdVBurgers) = u

function rhs_stiff!(du, u, equation::KdVBurgers, parameters, t)
    (; D2, D3, tmp) = parameters
    μ = equation.μ

    mul!(tmp, D2, u)
    mul!(du, D3, u)
    @. du = μ * tmp - du

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KdVBurgers, parameters)
    equation.μ * sparse(parameters.D2) - sparse(parameters.D3)
end

function rhs_nonstiff!(du, u, equation::KdVBurgers, parameters, t)
    (; D1, tmp) = parameters
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear invariant
    # (and the quadratic invariant for μ = 0)
    @. tmp = -one_third * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du - one_third * u * tmp

    return nothing
end

function dot_entropy(u, v, equation::KdVBurgers, parameters)
    (; D1, tmp) = parameters
    @. tmp = u * v
    return 0.5 * integrate(tmp, D1)
end

function setup(u_func, equation::KdVBurgers, tspan, D, D2 = nothing, D3 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3)
        D1 = D.central
        D2 = sparse(D.plus) * sparse(D.minus)
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
    elseif D isa PeriodicDerivativeOperator && D2 isa PeriodicDerivativeOperator && D3 isa PeriodicDerivativeOperator
        D1 = D
        D2 = D2
        D3 = D3
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    q0 = u0
    parameters = (; equation, D1, D2, D3, tmp)
    return (; q0, parameters)
end


#####################################################################
# Hyperbolized KdV-Burgers discretization

struct HyperbolizedKdVBurgers{T} <: AbstractEquation
    τ::T
    μ::T
end

get_u(q, equation::HyperbolizedKdVBurgers) = get_qi(q, equation, 0)
function get_qi(q, equation::HyperbolizedKdVBurgers, i)
    N = length(q) ÷ 3
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::HyperbolizedKdVBurgers, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0 * N + 1):(1 * N))
    dv = view(dq, (1 * N + 1):(2 * N))
    dw = view(dq, (2 * N + 1):(3 * N))

    u = view(q, (0 * N + 1):(1 * N))
    v = view(q, (1 * N + 1):(2 * N))
    w = view(q, (2 * N + 1):(3 * N))

    μ = equation.μ
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        # du .= -D₊ * w
        mul!(du, D1.plus, w, -1)

        # dv .= (D * v - w) / τ
        mul!(dv, D1.central, v)
        @. dv = inv_τ * (dv - w - μ * v)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1.minus, u)
        @. dw = inv_τ * (-dw + v)
    else
        # du .= -D₊ * w
        mul!(du, D1, w)
        @. du = -du

        # dv .= (D * v - w) / τ
        mul!(dv, D1, v)
        @. dv = inv_τ * (dv - w - μ * v)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1, u)
        @. dw = inv_τ * (-dw + v)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::HyperbolizedKdVBurgers, parameters)
    D1 = parameters.D1
    μ = equation.μ
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(D)
        jac = [O O -Dp;
               O inv_τ*(D-μ*I) -inv_τ*I;
               -inv_τ*Dm inv_τ*I O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*(D-μ*I) -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::HyperbolizedKdVBurgers, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * u^2
    mul!(du, D, tmp)
    mul!(tmp, D, u)
    @. du = du - one_third * u * tmp

    fill!(dv, zero(eltype(dv)))
    fill!(dw, zero(eltype(dw)))

    return nothing
end

function dot_entropy(q1, q2, equation::HyperbolizedKdVBurgers, parameters)
    (; D1, tmp) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0 * N + 1):(1 * N))
    v1 = view(q1, (1 * N + 1):(2 * N))
    w1 = view(q1, (2 * N + 1):(3 * N))

    u2 = view(q2, (0 * N + 1):(1 * N))
    v2 = view(q2, (1 * N + 1):(2 * N))
    w2 = view(q2, (2 * N + 1):(3 * N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp = half * (u1 * u2 + τ * v1 * v2 + τ * w1 * w2)

    return integrate(tmp, D1)
end


function setup(u_func, equation::HyperbolizedKdVBurgers, tspan, D1, D2 = nothing, D3 = nothing)
    if !isnothing(D2) || !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, KdVBurgers(equation.μ))

    if D1 isa PeriodicUpwindOperators
        v0 = D1.minus * u0 - equation.μ * u0
        w0 = D1.central * v0
    else
        v0 = D1 * u0 - equation.μ * u0
        w0 = D1 * v0
    end

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# Hyperbolized KdV-Burgers convergence test
function initial_condition(t, x::Number, equation::KdVBurgers)
    x0 = 25
    d = 5
    return 0.5 * (1 - tanh((abs(x) - x0) / d))
end
initial_condition(t, x::AbstractVector, equation::KdVBurgers) = initial_condition.(t, x, equation)

function korteweg_de_vries_burgers_convergence(; latex = false,
                                                 tspan = (0.0, 100.0),
                                                 N = 2^10,
                                                 accuracy_order = 7,
                                                 alg = ARS443(),
                                                 dt = 0.1,
                                                 kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -150.0
    xmax = 200.0
    μ = 0.1

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]

    u_limit, u_ini = let equation = KdVBurgers(μ)
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = HyperbolizedKdVBurgers(τ, μ)
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.central * q1_limit - μ * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical KdV-Burgers solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"KdV-Burgers solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"$|\!|(q_2 - \partial_x^2 u + \mu \partial_x u)(T\,)|\!|_{L^2}$")
    # Ideal convergence line
    ideal = @. τs / τs[end] * errors_u_limit[end]
    lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv; position = :lt, framevisible = false)

    filename = joinpath(FIGDIR, "korteweg_de_vries_burgers_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# Gardner discretization

struct Gardner{T} <: AbstractEquation
    σ::T
end

get_u(u, equations::Gardner) = u

function rhs_stiff!(du, u, equation::Gardner, parameters, t)
    (; D3) = parameters

    mul!(du, D3, u)
    @. du = -du

    return nothing
end

operator(::typeof(rhs_stiff!), equation::Gardner, parameters) = parameters.minus_D3

function rhs_nonstiff!(du, u, equation::Gardner, parameters, t)
    (; D1, tmp, tmp1) = parameters

    # This semidiscretization conserves the linear and quadratic invariants
    # σ u u_x
    factor = -equation.σ * one(eltype(u)) / 3
    @. tmp = factor * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du + factor * u * tmp
    # u^2 u_x
    factor = one(eltype(u)) / 6
    @. tmp1 = u^3
    mul!(tmp, D1, tmp1)
    @. du = du - factor * tmp
    @. tmp1 = u^2
    mul!(tmp, D1, tmp1)
    @. du = du - factor * u * tmp
    mul!(tmp, D1, u)
    @. du = du - factor * u^2 * tmp

    return nothing
end

function dot_entropy(u, v, equation::Gardner, parameters)
    (; D1, tmp) = parameters
    @. tmp = u * v
    return 0.5 * integrate(tmp, D1)
end

function setup(u_func, equation::Gardner, tspan, D, D3 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        minus_D3 = -D3
    elseif D isa PeriodicDerivativeOperator && D3 isa PeriodicDerivativeOperator
        D1 = D
        D3 = D3
        minus_D3 = -sparse(D3)
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    tmp1 = similar(u0)
    q0 = u0
    parameters = (; equation, D1, D3, minus_D3, tmp, tmp1)
    return (; q0, parameters)
end


#####################################################################
# Hyperbolized Gardner discretization

struct GardnerHyperbolized{T} <: AbstractEquation
    τ::T
    σ::T
end

get_u(q, equation::GardnerHyperbolized) = get_qi(q, equation, 0)
function get_qi(q, equation::GardnerHyperbolized, i)
    N = length(q) ÷ 3
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::GardnerHyperbolized, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0 * N + 1):(1 * N))
    dv = view(dq, (1 * N + 1):(2 * N))
    dw = view(dq, (2 * N + 1):(3 * N))

    u = view(q, (0 * N + 1):(1 * N))
    v = view(q, (1 * N + 1):(2 * N))
    w = view(q, (2 * N + 1):(3 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        # du .= -D₊ * w
        mul!(du, D1.plus, w, -1)

        # dv .= (D * v - w) / τ
        mul!(dv, D1.central, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1.minus, u)
        @. dw = inv_τ * (-dw + v)
    else
        # du .= -D₊ * w
        mul!(du, D1, w)
        @. du = -du

        # dv .= (D * v - w) / τ
        mul!(dv, D1, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1, u)
        @. dw = inv_τ * (-dw + v)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::GardnerHyperbolized, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(D)
        jac = [O O -Dp;
               O inv_τ*D -inv_τ*I;
               -inv_τ*Dm inv_τ*I O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*D -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::GardnerHyperbolized, parameters, t)
    (; D1, tmp, tmp1) = parameters
    N = size(D1, 2)

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    let u = u, du = du, D1 = D
        # This semidiscretization conserves the linear and quadratic invariants
        # σ u u_x
        factor = -equation.σ * one(eltype(u)) / 3
        @. tmp = factor * u^2
        mul!(du, D1, tmp)
        mul!(tmp, D1, u)
        @. du = du + factor * u * tmp
        # u^2 u_x
        factor = one(eltype(u)) / 6
        @. tmp1 = u^3
        mul!(tmp, D1, tmp1)
        @. du = du - factor * tmp
        @. tmp1 = u^2
        mul!(tmp, D1, tmp1)
        @. du = du - factor * u * tmp
        mul!(tmp, D1, u)
        @. du = du - factor * u^2 * tmp
    end

    fill!(dv, zero(eltype(dv)))
    fill!(dw, zero(eltype(dw)))

    return nothing
end

function dot_entropy(q1, q2, equation::GardnerHyperbolized, parameters)
    (; D1, tmp) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0 * N + 1):(1 * N))
    v1 = view(q1, (1 * N + 1):(2 * N))
    w1 = view(q1, (2 * N + 1):(3 * N))

    u2 = view(q2, (0 * N + 1):(1 * N))
    v2 = view(q2, (1 * N + 1):(2 * N))
    w2 = view(q2, (2 * N + 1):(3 * N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp = half * (u1 * u2 + τ * v1 * v2 + τ * w1 * w2)

    return integrate(tmp, D1)
end


function setup(u_func, equation::GardnerHyperbolized, tspan, D1, D3 = nothing)
    if !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, Gardner(equation.σ))

    if D1 isa PeriodicUpwindOperators
        v0 = D1.minus * u0
        w0 = D1.central * v0
    else
        v0 = D1 * u0
        w0 = D1 * v0
    end

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)
    tmp1 = similar(u0)

    parameters = (; equation, D1, tmp, tmp1)
    return (; q0, parameters)
end


#####################################################################
# Gardner convergence test
function solitary_wave(t, x::Number, equation::Gardner)
    xmin = -50.0
    xmax = +50.0

    c = 1.2
    A1 = 3 * c / sqrt(equation.σ^2 + 6 * c)
    A2 = (equation.σ / sqrt(equation.σ^2 + 6 * c) - 1) / 2
    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
    return A1 / (A2 + cosh(sqrt(c) / 2 * x_t)^2)
end
solitary_wave(t, x::AbstractVector, equation::Gardner) = solitary_wave.(t, x, equation)

function gardner_convergence(; latex = false,
                               domain_traversals = 1,
                               N = 50,
                               polydeg = 8,
                               alg = ARS443(),
                               dt = 0.01,
                               kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -50.0
    xmax = +50.0

    σ = 1.0
    c = 1.2
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    @show tspan

    D_local = legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = polydeg + 1)
    mesh = UniformPeriodicMesh1D(; xmin, xmax, Nx = N)
    Dm = couple_discontinuously(D_local, mesh, Val(:minus))
    D0 = couple_discontinuously(D_local, mesh, Val(:central))
    Dp = couple_discontinuously(D_local, mesh, Val(:plus))
    D1 = PeriodicUpwindOperators(Dm, D0, Dp)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]

    u_limit, u_ini = let equation = Gardner(σ)
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = GardnerHyperbolized(τ, σ)
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.central * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical Gardner solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"Gardner solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    # Ideal convergence line
    ideal = @. τs / τs[end] * errors_u_limit[end]
    lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv; position = :lt, framevisible = false)

    filename = joinpath(FIGDIR, "gardner_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# Linear bi-harmonic discretization

struct LinearBiharmonic <: AbstractEquation end

get_u(u, equations::LinearBiharmonic) = u

function rhs_stiff!(du, u, equation::LinearBiharmonic, parameters, t)
    (; D4) = parameters

    mul!(du, D4, u)
    @. du = -du

    return nothing
end

operator(::typeof(rhs_stiff!), equation::LinearBiharmonic, parameters) =
    -sparse(parameters.D4)

function rhs_nonstiff!(du, u, equation::LinearBiharmonic, parameters, t)
    fill!(du, zero(eltype(du)))
    return nothing
end

function setup(u_func, equation::LinearBiharmonic, tspan,
               D, D4 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D4)
        D1 = D.central
        D2 = sparse(D.plus) * sparse(D.minus)
        D4 = D2 * D2
    elseif D isa PeriodicDerivativeOperator && isnothing(D4)
        D1 = D
        D4 = D1^4
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    parameters = (; equation, D1, D4, tmp)
    return (; q0 = u0, parameters)
end


#####################################################################
# hyperbolized linear bi-harmonic discretization

struct LinearBiharmonicHyperbolized{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::LinearBiharmonicHyperbolized) = get_qi(q, equation, 0)
function get_qi(q, equation::LinearBiharmonicHyperbolized, i)
    N = length(q) ÷ 4
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::LinearBiharmonicHyperbolized, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    dq0 = view(dq, (0 * N + 1):(1 * N))
    dq1 = view(dq, (1 * N + 1):(2 * N))
    dq2 = view(dq, (2 * N + 1):(3 * N))
    dq3 = view(dq, (3 * N + 1):(4 * N))

    q0 = view(q, (0 * N + 1):(1 * N))
    q1 = view(q, (1 * N + 1):(2 * N))
    q2 = view(q, (2 * N + 1):(3 * N))
    q3 = view(q, (3 * N + 1):(4 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        mul!(dq0, D1.plus, q3)
        @. dq0 = -dq0

        mul!(dq1, D1.minus, q2)
        @. dq1 = inv_τ * (dq1 - q3)

        mul!(dq2, D1.plus, q1)
        @. dq2 = inv_τ * (dq2 - q2)

        mul!(dq3, D1.minus, q0)
        @. dq3 = -inv_τ * (dq3 - q1)
    else
        error()
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::LinearBiharmonicHyperbolized, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        Dp = sparse(D1.plus)
        O = zero(Dp)
        jac = [O O O -Dp;
               O O inv_τ*Dm -inv_τ*I;
               O inv_τ*Dp -inv_τ*I O;
               -inv_τ*Dm inv_τ*I O O]
        dropzeros!(jac)
        return jac
    else
        error()
    end
end

function rhs_nonstiff!(dq, q, equation::LinearBiharmonicHyperbolized, parameters, t)
    fill!(dq, zero(eltype(dq)))
    return nothing
end

function setup(u_func, equation::LinearBiharmonicHyperbolized, tspan, D1, D4 = nothing)
    if !isnothing(D4)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, LinearBiharmonic())

    if D1 isa PeriodicUpwindOperators
        q1 = D1.minus * u0
        q2 = D1.plus * q1
        q3 = D1.minus * q2
    else
        error()
    end

    q0 = vcat(u0, q1, q2, q3)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# Linear bi-harmonic convergence test
initial_condition(t, x::Number, equation::LinearBiharmonic) = exp(-t) * sin(x)
initial_condition(t, x::AbstractVector, equation::LinearBiharmonic) = initial_condition.(t, x, equation)

function linear_biharmonic_convergence(; latex = false,
                                         tspan = (0.0, 1.0),
                                         N = 2^5,
                                         accuracy_order = 3,
                                         alg = ARS443(),
                                         dt = 0.01,
                                         kwargs...)
    # Initialization of physical and numerical parameters
    xmin = 0.0
    xmax = 2 * π

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]
    errors_q3_limit = Float64[]

    u_limit, u_ini = let equation = LinearBiharmonic()
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = LinearBiharmonicHyperbolized(τ)
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.plus * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)

        q3_limit = parameters.D1.minus * q2_limit
        q3 = get_qi(sol.u[end], equation, 3)
        error_q3 = integrate(abs2, q3 - q3_limit, parameters.D1) |> sqrt
        push!(errors_q3_limit, error_q3)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical linear bi-harmomic solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"Linear bi-harmonic solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    ylims!(ax_sol, high = 1.5)
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    axislegend(ax_conv; position = :lt, framevisible = false)
    l3 = scatter!(ax_conv, τs, errors_q3_limit; label = L"Error $|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    l4 = lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv, [l3, l4], [L"$|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$", L"\propto \tau"]; position = :rb, framevisible = false)

    filename = joinpath(FIGDIR, "linear_biharmonic_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# Kuramoto-Sivashinsky discretization

struct KuramotoSivashinsky <: AbstractEquation end

get_u(u, equations::KuramotoSivashinsky) = u

function rhs_stiff!(du, u, equation::KuramotoSivashinsky, parameters, t)
    (; D2, D4, tmp) = parameters

    mul!(tmp, D2, u)
    mul!(du, D4, u)
    @. du = -du - tmp

    return nothing
end

operator(::typeof(rhs_stiff!), equation::KuramotoSivashinsky, parameters) =
    -sparse(parameters.D4) - sparse(parameters.D2)

function rhs_nonstiff!(du, u, equation::KuramotoSivashinsky, parameters, t)
    (; D1, tmp) = parameters
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear invariant
    @. tmp = -one_third * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du - one_third * u * tmp

    return nothing
end

function setup(u_func, equation::KuramotoSivashinsky, tspan,
               D, D2 = nothing, D4 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D2) && isnothing(D4)
        D1 = D.central
        D2 = sparse(D.plus) * sparse(D.minus)
        D4 = D2 * D2
    elseif D isa PeriodicDerivativeOperator && isnothing(D2) && isnothing(D4)
        D1 = D
        D2 = D1^2
        D4 = D1^4
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    parameters = (; equation, D1, D2, D4, tmp)
    return (; q0 = u0, parameters)
end


#####################################################################
# hyperbolized Kuramoto-Sivashinsky discretization

struct KuramotoSivashinskyHyperbolized{T} <: AbstractEquation
    τ::T
    use_q2::Bool
end

function KuramotoSivashinskyHyperbolized(τ::T, use_q2::Bool = true) where T
    return KuramotoSivashinskyHyperbolized{T}(τ, use_q2)
end

get_u(q, equation::KuramotoSivashinskyHyperbolized) = get_qi(q, equation, 0)
function get_qi(q, equation::KuramotoSivashinskyHyperbolized, i)
    N = length(q) ÷ 4
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::KuramotoSivashinskyHyperbolized, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    dq0 = view(dq, (0 * N + 1):(1 * N))
    dq1 = view(dq, (1 * N + 1):(2 * N))
    dq2 = view(dq, (2 * N + 1):(3 * N))
    dq3 = view(dq, (3 * N + 1):(4 * N))

    q0 = view(q, (0 * N + 1):(1 * N))
    q1 = view(q, (1 * N + 1):(2 * N))
    q2 = view(q, (2 * N + 1):(3 * N))
    q3 = view(q, (3 * N + 1):(4 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        mul!(dq0, D1.plus, q3)
        if equation.use_q2
            @. dq0 = -dq0 - q2
        else
            mul!(dq2, D1.plus, q1)
            @. dq0 = -dq0 - dq2
        end

        mul!(dq1, D1.minus, q2)
        @. dq1 = inv_τ * (dq1 - q3)

        mul!(dq2, D1.plus, q1)
        @. dq2 = inv_τ * (dq2 - q2)

        mul!(dq3, D1.minus, q0)
        @. dq3 = -inv_τ * (dq3 - q1)
    else
        error()
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KuramotoSivashinskyHyperbolized, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        Dp = sparse(D1.plus)
        O = zero(Dp)
        if equation.use_q2
            jac = [O O -I -Dp;
                   O O inv_τ*Dm -inv_τ*I;
                   O inv_τ*Dp -inv_τ*I O;
                   -inv_τ*Dm inv_τ*I O O]
        else
            jac = [O -Dp O -Dp;
                   O O inv_τ*Dm -inv_τ*I;
                   O inv_τ*Dp -inv_τ*I O;
                   -inv_τ*Dm inv_τ*I O O]
        end
        dropzeros!(jac)
        return jac
    else
        error()
    end
end

function rhs_nonstiff!(dq, q, equation::KuramotoSivashinskyHyperbolized, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    dq0 = view(dq, (0*N+1):(1*N))
    dq1 = view(dq, (1*N+1):(2*N))
    dq2 = view(dq, (2*N+1):(3*N))
    dq3 = view(dq, (3*N+1):(4*N))

    q0 = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * q0^2
    mul!(dq0, D, tmp)
    mul!(tmp, D, q0)
    @. dq0 = dq0 - one_third * q0 * tmp

    fill!(dq1, zero(eltype(dq1)))
    fill!(dq2, zero(eltype(dq2)))
    fill!(dq3, zero(eltype(dq3)))

    return nothing
end

function setup(u_func, equation::KuramotoSivashinskyHyperbolized, tspan, D1, D2 = nothing, D4 = nothing)
    if !isnothing(D2) || !isnothing(D4)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, KuramotoSivashinsky())

    if D1 isa PeriodicUpwindOperators
        q1 = D1.minus * u0
        q2 = D1.plus * q1
        q3 = D1.minus * q2
    else
        error()
    end

    q0 = vcat(u0, q1, q2, q3)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# Kuramoto-Sivashinsky convergence test
initial_condition(t, x::Number, equation::KuramotoSivashinsky) = exp(-x^2)
initial_condition(t, x::AbstractVector, equation::KuramotoSivashinsky) = initial_condition.(t, x, equation)

function kuramoto_sivashinsky_convergence(; latex = false,
                                            tspan = (0.0, 20.0),
                                            N = 2^8,
                                            accuracy_order = 7,
                                            alg = ARS443(),
                                            dt = 0.1,
                                            use_q2 = true,
                                            kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -50.0
    xmax = +50.0

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]
    errors_q3_limit = Float64[]

    u_limit, u_ini = let equation = KuramotoSivashinsky()
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = KuramotoSivashinskyHyperbolized(τ, use_q2)
        (; q0, parameters) = setup(initial_condition,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.plus * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)

        q3_limit = parameters.D1.minus * q2_limit
        q3 = get_qi(sol.u[end], equation, 3)
        error_q3 = integrate(abs2, q3 - q3_limit, parameters.D1) |> sqrt
        push!(errors_q3_limit, error_q3)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical Kuramoto-Sivashinsky solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"Kuramoto-Sivashinsky solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    ylims!(ax_sol, high = 3.2)
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    axislegend(ax_conv; position = :lt, framevisible = false)
    l3 = scatter!(ax_conv, τs, errors_q3_limit; label = L"Error $|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    l4 = lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv, [l3, l4], [L"$|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$", L"\propto \tau"]; position = :rb, framevisible = false)

    filename = joinpath(FIGDIR, "kuramoto_sivashinsky_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# Kawahara discretization

struct Kawahara <: AbstractEquation end

get_u(u, equations::Kawahara) = u

function rhs_stiff!(du, u, equation::Kawahara, parameters, t)
    (; D3, D5, tmp) = parameters

    mul!(tmp, D3, u)
    mul!(du, D5, u)
    @. du = du - tmp

    return nothing
end

operator(::typeof(rhs_stiff!), equation::Kawahara, parameters) =
    sparse(parameters.D5) - sparse(parameters.D3)

function rhs_nonstiff!(du, u, equation::Kawahara, parameters, t)
    (; D1, tmp) = parameters
    one_third = one(eltype(u)) / 3

    # This semidiscretization conserves the linear invariant
    @. tmp = -one_third * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du - one_third * u * tmp

    return nothing
end

function setup(u_func, equation::Kawahara, tspan,
               D, D3 = nothing, D5 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3) && isnothing(D5)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        D5 = sparse(D.plus) * D3 * sparse(D.minus)
    elseif D isa PeriodicDerivativeOperator && isnothing(D3) && isnothing(D5)
        D1 = D
        D3 = D1^3
        D5 = D1^5
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    parameters = (; equation, D1, D3, D5, tmp)
    return (; q0 = u0, parameters)
end


#####################################################################
# hyperbolized Kawahara discretization

struct KawaharaHyperbolized{T} <: AbstractEquation
    τ::T
end

get_u(q, equation::KawaharaHyperbolized) = get_qi(q, equation, 0)
function get_qi(q, equation::KawaharaHyperbolized, i)
    N = length(q) ÷ 5
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::KawaharaHyperbolized, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    dq0 = view(dq, (0 * N + 1):(1 * N))
    dq1 = view(dq, (1 * N + 1):(2 * N))
    dq2 = view(dq, (2 * N + 1):(3 * N))
    dq3 = view(dq, (3 * N + 1):(4 * N))
    dq4 = view(dq, (4 * N + 1):(5 * N))

    q0 = view(q, (0 * N + 1):(1 * N))
    q1 = view(q, (1 * N + 1):(2 * N))
    q2 = view(q, (2 * N + 1):(3 * N))
    q3 = view(q, (3 * N + 1):(4 * N))
    q4 = view(q, (4 * N + 1):(5 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        mul!(dq0, D1.plus, q4)

        mul!(dq1, D1.central, q1)
        mul!(dq3, D1.plus, q3)
        @. dq1 = -inv_τ * (dq3 - dq1 - q4)

        mul!(dq2, D1.central, q2)
        @. dq2 = inv_τ * (dq2 - q3)

        mul!(dq3, D1.minus, q1)
        @. dq3 = -inv_τ * (dq3 - q2)

        mul!(dq4, D1.minus, q0)
        @. dq4 = inv_τ * (dq4 - q1)
    else
        mul!(dq0, D1, q4)

        mul!(dq1, D1, q1)
        mul!(dq3, D1, q3)
        @. dq1 = -inv_τ * (dq3 - dq1 - q4)

        mul!(dq2, D1, q2)
        @. dq2 = inv_τ * (dq2 - q3)

        mul!(dq3, D1, q1)
        @. dq3 = -inv_τ * (dq3 - q2)

        mul!(dq4, D1, q0)
        @. dq4 = inv_τ * (dq4 - q1)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::KawaharaHyperbolized, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D0 = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(Dp)
        jac = [O O O O Dp;
               O inv_τ*D0 O -inv_τ*Dp inv_τ*I;
               O O inv_τ*D0 -inv_τ*I O;
               O -inv_τ*Dm inv_τ*I O O;
               inv_τ*Dm -inv_τ*I O O O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O O O D;
               O inv_τ*D O -inv_τ*D inv_τ*I;
               O O inv_τ*D -inv_τ*I O;
               O -inv_τ*D inv_τ*I O O;
               inv_τ*D -inv_τ*I O O O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::KawaharaHyperbolized, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    dq0 = view(dq, (0*N+1):(1*N))
    dq1 = view(dq, (1*N+1):(2*N))
    dq2 = view(dq, (2*N+1):(3*N))
    dq3 = view(dq, (3*N+1):(4*N))
    dq4 = view(dq, (4*N+1):(5*N))

    q0 = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * q0^2
    mul!(dq0, D, tmp)
    mul!(tmp, D, q0)
    @. dq0 = dq0 - one_third * q0 * tmp

    fill!(dq1, zero(eltype(dq1)))
    fill!(dq2, zero(eltype(dq2)))
    fill!(dq3, zero(eltype(dq3)))
    fill!(dq4, zero(eltype(dq4)))

    return nothing
end

function setup(u_func, equation::KawaharaHyperbolized, tspan, D1, D3 = nothing, D5 = nothing)
    if !isnothing(D3) || !isnothing(D5)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, Kawahara())

    if D1 isa PeriodicUpwindOperators
        q1 = D1.minus * u0
        q2 = D1.minus * q1
        q3 = D1.central * q2
        q4 = D1.plus * q3
    else
        q1 = D1 * u0
        q2 = D1 * q1
        q3 = D1 * q2
        q4 = D1 * q3
    end

    q0 = vcat(u0, q1, q2, q3, q4)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


#####################################################################
# Kawahara convergence test
function solitary_wave(t, x::Number, equation::Kawahara)
    xmin = -70.0
    xmax = +70.0

    c = 36 / 169
    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
    return 105 / (169 * cosh(x_t / (2 * sqrt(13)))^4)
end
solitary_wave(t, x::AbstractVector, equation::Kawahara) = solitary_wave.(t, x, equation)

function kawahara_convergence(; latex = false,
                                domain_traversals = 1,
                                N = 2^7,
                                accuracy_order = 3,
                                alg = ARS443(),
                                dt = 0.1,
                                kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -70.0
    xmax = +70.0

    c = 36 / 169
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    @show tspan

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]
    errors_q3_limit = Float64[]
    errors_q4_limit = Float64[]

    u_limit, u_ini = let equation = Kawahara()
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = KawaharaHyperbolized(τ)
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.minus * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)

        q3_limit = parameters.D1.central * q2_limit
        q3 = get_qi(sol.u[end], equation, 3)
        error_q3 = integrate(abs2, q3 - q3_limit, parameters.D1) |> sqrt
        push!(errors_q3_limit, error_q3)

        q4_limit = parameters.D1.plus * q3_limit - parameters.D1.central * q1_limit
        q4 = get_qi(sol.u[end], equation, 4)
        error_q4 = integrate(abs2, q4 - q4_limit, parameters.D1) |> sqrt
        push!(errors_q4_limit, error_q4)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical Kawahara solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"Kawahara solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    ylims!(ax_sol, high = 0.7)
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - \partial_x u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    axislegend(ax_conv; position = :lt, framevisible = false)
    l3 = scatter!(ax_conv, τs, errors_q3_limit; label = L"Error $|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$")
    l4 = scatter!(ax_conv, τs, errors_q4_limit; label = L"Error $|\!|q_4(T\,) - \partial_x^4 u(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    l5 = lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv, [l3, l4, l5], [L"$|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$", L"$|\!|q_4(T\,) - \partial_x^4 u(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$", L"\propto \tau"]; position = :rb, framevisible = false)

    filename = joinpath(FIGDIR, "kawahara_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end



#####################################################################
# Generalized Kawahara discretization

struct GeneralizedKawahara{T} <: AbstractEquation
    σ::T
end

get_u(u, equations::GeneralizedKawahara) = u

function rhs_stiff!(du, u, equation::GeneralizedKawahara, parameters, t)
    (; D3, D5, tmp) = parameters

    mul!(tmp, D3, u)
    mul!(du, D5, u)
    @. du = du - tmp

    return nothing
end

operator(::typeof(rhs_stiff!), equation::GeneralizedKawahara, parameters) =
    sparse(parameters.D5) - sparse(parameters.D3)

function rhs_nonstiff!(du, u, equation::GeneralizedKawahara, parameters, t)
    (; D1, tmp, tmp1) = parameters

    # This semidiscretization conserves the linear and quadratic invariants
    # σ u u_x
    factor = -equation.σ * one(eltype(u)) / 3
    @. tmp = factor * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du + factor * u * tmp
    # u^2 u_x
    factor = one(eltype(u)) / 6
    @. tmp1 = u^3
    mul!(tmp, D1, tmp1)
    @. du = du - factor * tmp
    @. tmp1 = u^2
    mul!(tmp, D1, tmp1)
    @. du = du - factor * u * tmp
    mul!(tmp, D1, u)
    @. du = du - factor * u^2 * tmp

    return nothing
end

function dot_entropy(u, v, equation::GeneralizedKawahara, parameters)
    (; D1, tmp) = parameters
    @. tmp = u * v
    return integrate(tmp, D1)
end

function setup(u_func, equation::GeneralizedKawahara, tspan,
               D, D3 = nothing, D5 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3) && isnothing(D5)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        D5 = sparse(D.plus) * D3 * sparse(D.minus)
    elseif D isa PeriodicDerivativeOperator && isnothing(D3) && isnothing(D5)
        D1 = D
        D3 = D1^3
        D5 = D1^5
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    tmp1 = similar(u0)
    parameters = (; equation, D1, D3, D5, tmp, tmp1)
    return (; q0 = u0, parameters)
end


#####################################################################
# hyperbolized generalized Kawahara discretization

struct GeneralizedKawaharaHyperbolized{T} <: AbstractEquation
    τ::T
    σ::T
end

get_u(q, equation::GeneralizedKawaharaHyperbolized) = get_qi(q, equation, 0)
function get_qi(q, equation::GeneralizedKawaharaHyperbolized, i)
    N = length(q) ÷ 5
    return view(q, (i * N + 1):((i + 1) * N))
end

function rhs_stiff!(dq, q, equation::GeneralizedKawaharaHyperbolized, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    dq0 = view(dq, (0 * N + 1):(1 * N))
    dq1 = view(dq, (1 * N + 1):(2 * N))
    dq2 = view(dq, (2 * N + 1):(3 * N))
    dq3 = view(dq, (3 * N + 1):(4 * N))
    dq4 = view(dq, (4 * N + 1):(5 * N))

    q0 = view(q, (0 * N + 1):(1 * N))
    q1 = view(q, (1 * N + 1):(2 * N))
    q2 = view(q, (2 * N + 1):(3 * N))
    q3 = view(q, (3 * N + 1):(4 * N))
    q4 = view(q, (4 * N + 1):(5 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        mul!(dq0, D1.plus, q4)

        mul!(dq1, D1.central, q1)
        mul!(dq3, D1.plus, q3)
        @. dq1 = -inv_τ * (dq3 - dq1 - q4)

        mul!(dq2, D1.central, q2)
        @. dq2 = inv_τ * (dq2 - q3)

        mul!(dq3, D1.minus, q1)
        @. dq3 = -inv_τ * (dq3 - q2)

        mul!(dq4, D1.minus, q0)
        @. dq4 = inv_τ * (dq4 - q1)
    else
        mul!(dq0, D1, q4)

        mul!(dq1, D1, q1)
        mul!(dq3, D1, q3)
        @. dq1 = -inv_τ * (dq3 - dq1 - q4)

        mul!(dq2, D1, q2)
        @. dq2 = inv_τ * (dq2 - q3)

        mul!(dq3, D1, q1)
        @. dq3 = -inv_τ * (dq3 - q2)

        mul!(dq4, D1, q0)
        @. dq4 = inv_τ * (dq4 - q1)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::GeneralizedKawaharaHyperbolized, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D0 = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(Dp)
        jac = [O O O O Dp;
               O inv_τ*D0 O -inv_τ*Dp inv_τ*I;
               O O inv_τ*D0 -inv_τ*I O;
               O -inv_τ*Dm inv_τ*I O O;
               inv_τ*Dm -inv_τ*I O O O]
        dropzeros!(jac)
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O O O D;
               O inv_τ*D O -inv_τ*D inv_τ*I;
               O O inv_τ*D -inv_τ*I O;
               O -inv_τ*D inv_τ*I O O;
               inv_τ*D -inv_τ*I O O O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::GeneralizedKawaharaHyperbolized, parameters, t)
    (; D1, tmp, tmp1) = parameters
    N = size(D1, 2)

    dq0 = view(dq, (0*N+1):(1*N))
    dq1 = view(dq, (1*N+1):(2*N))
    dq2 = view(dq, (2*N+1):(3*N))
    dq3 = view(dq, (3*N+1):(4*N))
    dq4 = view(dq, (4*N+1):(5*N))

    q0 = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    let u = q0, du = dq0, D1 = D
        # This semidiscretization conserves the linear and quadratic invariants
        # σ u u_x
        factor = -equation.σ * one(eltype(u)) / 3
        @. tmp = factor * u^2
        mul!(du, D1, tmp)
        mul!(tmp, D1, u)
        @. du = du + factor * u * tmp
        # u^2 u_x
        factor = one(eltype(u)) / 6
        @. tmp1 = u^3
        mul!(tmp, D1, tmp1)
        @. du = du - factor * tmp
        @. tmp1 = u^2
        mul!(tmp, D1, tmp1)
        @. du = du - factor * u * tmp
        mul!(tmp, D1, u)
        @. du = du - factor * u^2 * tmp
    end

    fill!(dq1, zero(eltype(dq1)))
    fill!(dq2, zero(eltype(dq2)))
    fill!(dq3, zero(eltype(dq3)))
    fill!(dq4, zero(eltype(dq4)))

    return nothing
end

function dot_entropy(qa, qb, equation::GeneralizedKawaharaHyperbolized, parameters)
    (; D1, tmp) = parameters
    (; τ) = equation
    N = size(D1, 2)

    q0a = view(qa, (0*N+1):(1*N))
    q1a = view(qa, (1*N+1):(2*N))
    q2a = view(qa, (2*N+1):(3*N))
    q3a = view(qa, (3*N+1):(4*N))
    q4a = view(qa, (4*N+1):(5*N))

    q0b = view(qb, (0*N+1):(1*N))
    q1b = view(qb, (1*N+1):(2*N))
    q2b = view(qb, (2*N+1):(3*N))
    q3b = view(qb, (3*N+1):(4*N))
    q4b = view(qb, (4*N+1):(5*N))

    @. tmp = q0a * q0b + τ * (q1a * q1b + q2a * q2b + q3a * q3b + q4a * q4b)

    return integrate(tmp, D1)
end

function setup(u_func, equation::GeneralizedKawaharaHyperbolized, tspan, D1, D3 = nothing, D5 = nothing)
    if !isnothing(D3) || !isnothing(D5)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = grid(D1)
    u0 = u_func(tspan[1], x, GeneralizedKawahara(equation.σ))

    if D1 isa PeriodicUpwindOperators
        q1 = D1.minus * u0
        q2 = D1.minus * q1
        q3 = D1.central * q2
        q4 = D1.plus * q3
    else
        q1 = D1 * u0
        q2 = D1 * q1
        q3 = D1 * q2
        q4 = D1 * q3
    end

    q0 = vcat(u0, q1, q2, q3, q4)

    tmp = similar(u0)
    tmp1 = similar(u0)

    parameters = (; equation, D1, tmp, tmp1)
    return (; q0, parameters)
end


#####################################################################
# Generalized Kawahara convergence test
function solitary_wave(t, x::Number, equation::GeneralizedKawahara)
    xmin = -70.0
    xmax = +70.0

    k = sqrt(1 / 20 + equation.σ / (4 * sqrt(10)))
    c = 4 * k^2 * (1 - 4 * k^2)
    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
    return -6 * sqrt(10) * k^2 / cosh(k * x_t)^2
end
solitary_wave(t, x::AbstractVector, equation::GeneralizedKawahara) = solitary_wave.(t, x, equation)

function generalized_kawahara_convergence(; latex = false,
                                            domain_traversals = 1,
                                            N = 2^7,
                                            accuracy_order = 7,
                                            alg = ARS443(),
                                            dt = 0.1,
                                            kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -70.0
    xmax = +70.0

    σ = 2 / sqrt(90)
    k = sqrt(1 / 20 + σ / (4 * sqrt(10)))
    c = 4 * k^2 * (1 - 4 * k^2)
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    @show tspan

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    τs = 10.0 .^ (-1:-1:-12)
    errors_u_limit = Float64[]
    errors_q1_limit = Float64[]
    errors_q2_limit = Float64[]
    errors_q3_limit = Float64[]
    errors_q4_limit = Float64[]

    u_limit, u_ini = let equation = GeneralizedKawahara(σ)
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation), get_u(sol.u[begin], equation)
    end

    for τ in τs
        equation = GeneralizedKawaharaHyperbolized(τ, σ)
        (; q0, parameters) = setup(solitary_wave,
                                   equation, tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        u = get_u(sol.u[end], equation)
        error_u = integrate(abs2, u - u_limit, parameters.D1) |> sqrt
        push!(errors_u_limit, error_u)

        q1_limit = parameters.D1.minus * u_limit
        q1 = get_qi(sol.u[end], equation, 1)
        error_q1 = integrate(abs2, q1 - q1_limit, parameters.D1) |> sqrt
        push!(errors_q1_limit, error_q1)

        q2_limit = parameters.D1.minus * q1_limit
        q2 = get_qi(sol.u[end], equation, 2)
        error_q2 = integrate(abs2, q2 - q2_limit, parameters.D1) |> sqrt
        push!(errors_q2_limit, error_q2)

        q3_limit = parameters.D1.central * q2_limit
        q3 = get_qi(sol.u[end], equation, 3)
        error_q3 = integrate(abs2, q3 - q3_limit, parameters.D1) |> sqrt
        push!(errors_q3_limit, error_q3)

        q4_limit = parameters.D1.plus * q3_limit - parameters.D1.central * q1_limit
        q4 = get_qi(sol.u[end], equation, 4)
        error_q4 = integrate(abs2, q4 - q4_limit, parameters.D1) |> sqrt
        push!(errors_q4_limit, error_q4)
    end

    let errors_u = errors_u_limit
        @info "Errors with respect to the numerical generalized Kawahara solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = [L"\tau", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend = Val(:latex))
        end
    end

    fig = Figure(size = (1200, 450)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x")
    lines!(ax_sol, grid(D1), u_limit; label = L"Generalized Kawahara solution $u$")
    lines!(ax_sol, grid(D1), u_ini; label = L"initial data $u_0$")
    ylims!(ax_sol, high = 0.7)
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_conv = Axis(fig[1, 2];
                   xlabel = L"Relaxation parameter $\tau$",
                   xscale = log10, yscale = log10)
    scatter!(ax_conv, τs, errors_u_limit; label = L"Error $|\!|q_0(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q1_limit; label = L"Error $|\!|q_1(T\,) - u(T\,)|\!|_{L^2}$")
    scatter!(ax_conv, τs, errors_q2_limit; label = L"Error $|\!|q_2(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    axislegend(ax_conv; position = :lt, framevisible = false)
    l3 = scatter!(ax_conv, τs, errors_q3_limit; label = L"Error $|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$")
    l4 = scatter!(ax_conv, τs, errors_q4_limit; label = L"Error $|\!|q_4(T\,) - \partial_x^4 u(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$")
    ideal = @. τs / τs[end] * errors_u_limit[end]
    l5 = lines!(ax_conv, τs, ideal; label = L"\propto \tau", color = :gray, linestyle = :dot)
    axislegend(ax_conv, [l3, l4, l5], [L"$|\!|q_3(T\,) - \partial_x^3 u(T\,)|\!|_{L^2}$", L"$|\!|q_4(T\,) - \partial_x^4 u(T\,) - \partial_x^2 u(T\,)|\!|_{L^2}$", L"\propto \tau"]; position = :rb, framevisible = false)

    filename = joinpath(FIGDIR, "generalized_kawahara_convergence.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end

function generalized_kawahara_error_growth(; domain_traversals = 10,
                                             N = 2^9,
                                             accuracy_order = 7,
                                             alg = ARS443(),
                                             dt = 0.1,
                                             kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -70.0
    xmax = +70.0

    σ = 2 / sqrt(90)
    k = sqrt(1 / 20 + σ / (4 * sqrt(10)))
    c = 4 * k^2 * (1 - 4 * k^2)
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    @show tspan

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    # Setup callback computing the error
    times = Float64[]
    errors = Float64[]
    callback = let times = times, errors = errors
        function (q, parameters, t)
            (; tmp, equation) = parameters

            u = get_u(q, equation)
            tmp .= solitary_wave.(t, grid(parameters.D1), GeneralizedKawahara(σ))

            @. tmp = u - tmp
            err = integrate(abs2, tmp, parameters.D1) |> sqrt

            push!(times, t)
            push!(errors, err)
            return nothing
        end
    end

    times_baseline_standard = similar(times)
    times_baseline_relaxation = similar(times)
    errors_baseline_standard = similar(errors)
    errors_baseline_relaxation = similar(errors)

    let equation = GeneralizedKawahara(σ)
        (; q0, parameters) = setup(solitary_wave, equation, tspan, D1)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, kwargs...)
        resize!(times_baseline_standard, length(times))
        resize!(errors_baseline_standard, length(errors))
        copyto!(times_baseline_standard, times)
        copyto!(errors_baseline_standard, errors)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, relaxation = true, kwargs...)
        resize!(times_baseline_relaxation, length(times))
        resize!(errors_baseline_relaxation, length(errors))
        copyto!(times_baseline_relaxation, times)
        copyto!(errors_baseline_relaxation, errors)
    end

    fig = Figure(size = (1200, 900)) # default size is (600, 450)

    ax1 = Axis(fig[1, 1];
               title = L"\tau = 10^{-3}",
               xlabel = L"Time $t$",
               ylabel = L"Error $|\!|u - u_\mathrm{ana}|\!|_{L^2}$",
               xscale = log10, yscale = log10)
    lines!(ax1, times_baseline_standard, errors_baseline_standard; label = "baseline, standard")
    lines!(ax1, times_baseline_relaxation, errors_baseline_relaxation; label = "baseline, relaxation")
    let equation = GeneralizedKawaharaHyperbolized(1.0e-3, σ)
        (; q0, parameters) = setup(solitary_wave, equation, tspan, D1)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, kwargs...)
        lines!(ax1, times, errors; label = "hyperbolization, standard")

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, relaxation = true, kwargs...)
        lines!(ax1, times, errors; label = "hyperbolization, relaxation")
    end
    axislegend(ax1; position = :lt, framevisible = false)

    ax2 = Axis(fig[1, 2];
               title = L"\tau = 10^{-4}",
               xlabel = L"Time $t$",
               ylabel = L"Error $|\!|u - u_\mathrm{ana}|\!|_{L^2}$",
               xscale = log10, yscale = log10)
    lines!(ax2, times_baseline_standard, errors_baseline_standard; label = "baseline, standard")
    lines!(ax2, times_baseline_relaxation, errors_baseline_relaxation; label = "baseline, relaxation")
    let equation = GeneralizedKawaharaHyperbolized(1.0e-4, σ)
        (; q0, parameters) = setup(solitary_wave, equation, tspan, D1)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, kwargs...)
        lines!(ax2, times, errors; label = "hyperbolization, standard")

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, relaxation = true, kwargs...)
        lines!(ax2, times, errors; label = "hyperbolization, relaxation")
    end
    axislegend(ax2; position = :lt, framevisible = false)

    ax3 = Axis(fig[2, 1];
               title = L"\tau = 10^{-5}",
               xlabel = L"Time $t$",
               ylabel = L"Error $|\!|u - u_\mathrm{ana}|\!|_{L^2}$",
               xscale = log10, yscale = log10)
    lines!(ax3, times_baseline_standard, errors_baseline_standard; label = "baseline, standard")
    lines!(ax3, times_baseline_relaxation, errors_baseline_relaxation; label = "baseline, relaxation")
    let equation = GeneralizedKawaharaHyperbolized(1.0e-5, σ)
        (; q0, parameters) = setup(solitary_wave, equation, tspan, D1)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, kwargs...)
        lines!(ax3, times, errors; label = "hyperbolization, standard")

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, relaxation = true, kwargs...)
        lines!(ax3, times, errors; label = "hyperbolization, relaxation")
    end
    axislegend(ax3; position = :lt, framevisible = false)

    ax4 = Axis(fig[2, 2];
               title = L"\tau = 10^{-6}",
               xlabel = L"Time $t$",
               ylabel = L"Error $|\!|u - u_\mathrm{ana}|\!|_{L^2}$",
               xscale = log10, yscale = log10)
    lines!(ax4, times_baseline_standard, errors_baseline_standard; label = "baseline, standard")
    lines!(ax4, times_baseline_relaxation, errors_baseline_relaxation; label = "baseline, relaxation")
    let equation = GeneralizedKawaharaHyperbolized(1.0e-6, σ)
        (; q0, parameters) = setup(solitary_wave, equation, tspan, D1)

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, kwargs...)
        lines!(ax4, times, errors; label = "hyperbolization, standard")

        empty!(times)
        empty!(errors)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, callback, relaxation = true, kwargs...)
        lines!(ax4, times, errors; label = "hyperbolization, relaxation")
    end
    axislegend(ax4; position = :lt, framevisible = false)

    filename = joinpath(FIGDIR, "generalized_kawahara_error_growth.pdf")
    save(filename, fig)
    @info "Results saved to $filename"

    return nothing
end

