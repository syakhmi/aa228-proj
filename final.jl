using Parameters
using Random

using POMDPs, POMDPSimulators, POMDPPolicies
using MCTS
using BasicPOMCP

import POMDPs: initialstate_distribution, action, actions, gen, discount, isterminal, rand, updater

@with_kw mutable struct State
    p1::Vector{Int64}                       # Player 1's hand
    p1_d::Vector{Int64} = Vector{Int64}()   # Player 1's discarded cards
    p2::Vector{Int64}                       # Player 2's hand
    p2_d::Vector{Int64} = Vector{Int64}()   # Player 2's discarded cards
    p3::Vector{Int64}                       # Player 3's hand
    p3_d::Vector{Int64} = Vector{Int64}()   # Player 3's discarded cards
    p4::Vector{Int64}                       # Player 4's hand
    p4_d::Vector{Int64} = Vector{Int64}()   # Player 4's discarded cards
    p1_w::Int64 = 0                         # Number of tricks won by Player 1
    p2_w::Int64 = 0                         # Number of tricks won by Player 2
    p3_w::Int64 = 0                         # Number of tricks won by Player 3
    p4_w::Int64 = 0                         # Number of tricks won by Player 4
    t::Int64 = 0                            # Timestep
end

Base.copy(s::State) = State(
        p1=s.p1,
        p1_d=s.p1_d,
        p2=s.p2,
        p2_d=s.p2_d,
        p3=s.p3,
        p3_d=s.p3_d,
        p4=s.p4,
        p4_d=s.p4_d,
        p1_w=s.p1_w,
        p2_w=s.p2_w,
        p3_w=s.p3_w,
        p4_w=s.p4_w,
        t=s.t,
    )

@enum Suite clubs diamonds hearts spades

function GetSuite(c::Int64)
    if c <= 13
        return clubs
    elseif c <= 13 * 2
        return diamonds
    elseif c <= 13 * 3
        return hearts
    else
        return spades
    end
end

function SuiteToString(s::Suite)
    if s == clubs
        return "C"
    elseif s == diamonds
        return "D"
    elseif s == hearts
        return "H"
    else
        return "S"
    end
end

function CardToString(c::Int64)
    digit = c % 13
    digit = digit == 0 ? 13 : digit
    return "$(digit)$(SuiteToString(GetSuite(c)))"
end

function Base.show(io::IO, s::State)
    println(io, "State(t=$(s.t), p1_w=$(s.p1_w), p2_w=$(s.p2_w), p3_w=$(s.p3_w), p4_w=$(s.p4_w))")
    println(io, "  Player 1 Hand: $([CardToString(c) for c in s.p1])")
    println(io, "  Player 2 Hand: $([CardToString(c) for c in s.p2])")
    println(io, "  Player 3 Hand: $([CardToString(c) for c in s.p3])")
    println(io, "  Player 4 Hand: $([CardToString(c) for c in s.p4])")
    println(io, "  Player 1 Discard: $([CardToString(c) for c in s.p1_d])")
    println(io, "  Player 2 Discard: $([CardToString(c) for c in s.p2_d])")
    println(io, "  Player 3 Discard: $([CardToString(c) for c in s.p3_d])")
    println(io, "  Player 4 Discard: $([CardToString(c) for c in s.p4_d])")
end

@with_kw mutable struct Observation
    p1::Vector{Int64}   # Player 1's hand
    p1_d::Vector{Int64} # Player 1's discarded cards
    p2_d::Vector{Int64} # Player 2's discarded cards
    p3_d::Vector{Int64} # Player 3's discarded cards
    p4_d::Vector{Int64} # Player 4's discarded cards
    p1_w::Int64         # Number of tricks won by Player 1
    p2_w::Int64         # Number of tricks won by Player 2
    p3_w::Int64         # Number of tricks won by Player 3
    p4_w::Int64         # Number of tricks won by Player 4
    t::Int64
end

@with_kw mutable struct Action
    n::Int64  # The card to play
end

@with_kw mutable struct Contract
    suite::Suite
    n::Int64
    p::Int64
end

@with_kw struct PostBiddingProblem <: POMDP{State, Action, Observation}
    contract::Contract
    n_cards::Int64
end

function TrickSortLt(trump::Suite)
    function Inner(x::Int64, y::Int64)
        x_suite = GetSuite(x)
        y_suite = GetSuite(y)

        if x_suite == y_suite
            return x < y
        end

        if x_suite == trump
            return false
        end

        if y_suite == trump
            return true
        end

        return x_suite < y_suite
    end

    return Inner
end

function TrickSortLt(trump::Suite, first_suite::Suite)
    function Inner(x::Int64, y::Int64)
        x_suite = GetSuite(x)
        y_suite = GetSuite(y)

        if x_suite == y_suite
            return x < y
        end

        if x_suite == trump
            return false
        end

        if y_suite == trump
            return true
        end

        if x_suite == first_suite
            return false
        end

        if y_suite == first_suite
            return true
        end

        return x_suite < y_suite
    end

    return Inner
end

function GetPlayed(
    p1_d::Vector{Int64},
    p2_d::Vector{Int64},
    p3_d::Vector{Int64},
    p4_d::Vector{Int64},
    pn::Int64,
    t::Int64)

    p1_c = length(p1_d) == t ? p1_d[t] : nothing
    p2_c = length(p2_d) == t ? p2_d[t] : nothing
    p3_c = length(p3_d) == t ? p3_d[t] : nothing
    p4_c = length(p4_d) == t ? p4_d[t] : nothing

    played_nothing = [p1_c, p2_c, p3_c, p4_c]

    first_card = nothing
    for i in 0:3
        idx = pn - i - 1
        idx = idx < 0 ? idx + 4 : idx
        idx += 1

        c = played_nothing[idx]
        if idx == pn && c == nothing
            continue
        elseif c == nothing
            break
        end
        first_card = c
    end

    played = Vector{Int64}()
    for c in played_nothing
        if c != nothing
            push!(played, c)
        end
    end

    return played, first_card
end

function GreedyAction(
    m::PostBiddingProblem,
    pn_hand::Vector{Int64},
    p1_d::Vector{Int64},
    p2_d::Vector{Int64},
    p3_d::Vector{Int64},
    p4_d::Vector{Int64},
    pn::Int64,
    t::Int64)
    if pn == 1
        partner_card = length(p3_d) == t ? p3_d[t] : -1
    elseif pn == 2
        partner_card = length(p4_d) == t ? p4_d[t] : -1
    elseif pn == 3
        partner_card = length(p1_d) == t ? p1_d[t] : -1
    else
        partner_card = length(p2_d) == t ? p2_d[t] : -1
    end

    played, first_card = GetPlayed(p1_d, p2_d, p3_d, p4_d, pn, t)

    # If no one has played, duck
    if length(played) == 0
        return Action(n=1)
    end

    trick_lt = TrickSortLt(m.contract.suite, GetSuite(first_card))

    sort!(played, lt=trick_lt, rev=true)
    best_card = played[1]
    partner_winning = partner_card == best_card

    # If partner has the trick, duck
    if partner_winning
        return Action(n=1)
    end

    # Play lowest value card that can win trick
    for c in 1:length(pn_hand)
        if trick_lt(best_card, pn_hand[c])
            return Action(n=c)
        end
    end

    # Can't win, so duck
    return Action(n=1)
end

function TrickWinner(m::PostBiddingProblem, s::State, last_player::Int64)
    played, first_card = GetPlayed(s.p1_d, s.p2_d, s.p3_d, s.p4_d, last_player, s.t)

    trick_lt = TrickSortLt(m.contract.suite, GetSuite(first_card))

    return sortperm(played, lt=trick_lt, rev=true)[1]
end

function IsOver(s::State)
    last_round = s.t > 13
    p1_done = length(s.p1_d) == 13
    p2_done = length(s.p2_d) == 13
    p3_done = length(s.p3_d) == 13
    p4_done = length(s.p4_d) == 13
    return last_round && p1_done && p2_done && p3_done && p4_done
end

function ExtractObs(s::State)
    return Observation(
        p1=s.p1,
        p1_d=s.p1_d,
        p2_d=s.p2_d,
        p3_d=s.p3_d,
        p4_d=s.p4_d,
        p1_w=s.p1_w,
        p2_w=s.p2_w,
        p3_w=s.p3_w,
        p4_w=s.p4_w,
        t=s.t,
    )
end

function Reward(m::PostBiddingProblem, s::State)
    if !IsOver(s)
        return 0
    end

    wins = s.p1_w + s.p3_w

    if wins - m.contract.n >= 0
        return 2 * m.contract.n + wins - m.contract.n
    else
        return 2 * (wins - m.contract.n)
    end
end

function gen(m::PostBiddingProblem, s::State, a::Action, rng::AbstractRNG)
    if a.n > m.n_cards - s.t + 1
        # Invalid Action! Shouldn't happen.
        return (sp=s, o=ExtractObs(s), r=-100)
    end

    sp = copy(s)

    # Check if starting new round

    if length(sp.p1_d) == sp.t && length(sp.p2_d) == sp.t && length(sp.p3_d) == sp.t && length(sp.p4_d) == sp.t
        sp.t += 1
    end

    # Finish Current Round


    # Player 1 Turn
    card_to_play = sp.p1[a.n]
    sp.p1 = sp.p1[1:end .!= a.n]
    sp.p1_d = [sp.p1_d; card_to_play]
    last_player_of_round = 1

    # Player 2 Turn
    if length(sp.p2_d) < sp.t
        p2_a = GreedyAction(m, sp.p2, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 2, sp.t)
        card_to_play = sp.p2[p2_a.n]
        sp.p2 = sp.p2[1:end .!= p2_a.n]
        sp.p2_d = [sp.p2_d; card_to_play]
        last_player_of_round = 2
    end

    # Player 3 Turn
    if length(sp.p3_d) < sp.t
        p3_a = GreedyAction(m, sp.p3, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 3, sp.t)
        card_to_play = sp.p3[p3_a.n]
        sp.p3 = sp.p3[1:end .!= p3_a.n]
        sp.p3_d = [sp.p3_d; card_to_play]
        last_player_of_round = 3
    end

    # Player 4 Turn
    if length(sp.p4_d) < sp.t
        p4_a = GreedyAction(m, sp.p4, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 4, sp.t)
        card_to_play = sp.p4[p4_a.n]
        sp.p4 = sp.p4[1:end .!= p4_a.n]
        sp.p4_d = [sp.p4_d; card_to_play]
        last_player_of_round = 4
    end

    winner = TrickWinner(m, sp, last_player_of_round)

    if winner == 1
        sp.p1_w += 1
    elseif winner == 2
        sp.p2_w += 1
    elseif winner == 3
        sp.p3_w += 1
    else
        sp.p4_w += 1
    end

    sp.t += 1

    if winner == 1 || IsOver(sp)
        return (sp=sp, o=ExtractObs(sp), r=Reward(m, sp))
    end

    # Start next round if P1 did not win trick and game is not over

    if winner <= 2
        p2_a = GreedyAction(m, sp.p2, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 2, sp.t)
        card_to_play = sp.p2[p2_a.n]
        sp.p2 = sp.p2[1:end .!= p2_a.n]
        sp.p2_d = [sp.p2_d; card_to_play]
    end

    if winner <= 3
        p3_a = GreedyAction(m, sp.p3, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 3, sp.t)
        card_to_play = sp.p3[p3_a.n]
        sp.p3 = sp.p3[1:end .!= p3_a.n]
        sp.p3_d = [sp.p3_d; card_to_play]
    end

    if winner <= 4
        p4_a = GreedyAction(m, sp.p4, sp.p1_d, sp.p2_d, sp.p3_d, sp.p4_d, 4, sp.t)
        card_to_play = sp.p4[p4_a.n]
        sp.p4 = sp.p4[1:end .!= p4_a.n]
        sp.p4_d = [sp.p4_d; card_to_play]
    end

    return (sp=sp, o=ExtractObs(sp), r=Reward(m, sp))
end

struct ObservationDistribution
    m::PostBiddingProblem
    o::Union{Observation, Nothing}
end

actions(::PostBiddingProblem) = [Action(n=x) for x in 1:13]

function actions(::PostBiddingProblem, b::ObservationDistribution)
    o = b.o
    if o == nothing
        return [Action(n=x) for x in 1:13]
    end
    return [Action(n=x) for x in 1:length(o.p1)]
end

function actions(::PostBiddingProblem, b)
    o = first(b.hist)[:o]
    if o == nothing
        return [Action(n=x) for x in 1:13]
    end
    return [Action(n=x) for x in 1:length(o.p1)]
end

discount(m::PostBiddingProblem) = 1

function isterminal(m::PostBiddingProblem, s::State)
    return IsOver(s)
end

initialstate_distribution(m::PostBiddingProblem) = ObservationDistribution(m, nothing)

function rand(rng::AbstractRNG, d::ObservationDistribution)
    if d.o == nothing
        deck = vec(collect(1:52))
        shuffle!(rng, deck)
        p1 = deck[1:13]
        p2 = deck[13+1:13*2]
        p3 = deck[13*2+1:13*3]
        p4 = deck[13*3+1:end]

        p1_d = Vector{Int64}()
        p2_d = Vector{Int64}()
        p3_d = Vector{Int64}()
        p4_d = Vector{Int64}()

        p1_w = 0
        p2_w = 0
        p3_w = 0
        p4_w = 0

        t = 0
    else
        p1 = d.o.p1

        played = vcat(p1, d.o.p2_d, d.o.p3_d, d.o.p4_d)
        deck = vec(collect(1:52))
        for i in played
            deck[i] = -1
        end
        deck = deck[1:end .> 0]

        shuffle!(rng, deck)
        p2_n = 13 - length(d.o.p2_d)
        p3_n = 13 - length(d.o.p3_d)
        p4_n = 13 - length(d.o.p4_d)

        p2 = deck[1:p2_n]
        p3 = deck[p2_n+1:p2_n+p3_n]
        p4 = deck[p2_n+p3_n+1:end]

        p1_d = d.o.p1_d
        p2_d = d.o.p2_d
        p3_d = d.o.p3_d
        p4_d = d.o.p4_d

        p1_w = d.o.p1_w
        p2_w = d.o.p2_w
        p3_w = d.o.p3_w
        p4_w = d.o.p4_w

        t = d.o.t
    end

    trick_lt = TrickSortLt(d.m.contract.suite)

    sort!(p1, lt=trick_lt)
    sort!(p2, lt=trick_lt)
    sort!(p3, lt=trick_lt)
    sort!(p4, lt=trick_lt)

    s = State(
        p1=p1,
        p2=p2,
        p3=p3,
        p4=p4,
        p1_d=p1_d,
        p2_d=p2_d,
        p3_d=p3_d,
        p4_d=p4_d,
        p1_w=p1_w,
        p2_w=p2_w,
        p3_w=p3_w,
        p4_w=p4_w,
        t=t,
    )

    return s
end

struct GreedyPolicy <: Policy
    m::PostBiddingProblem
end

struct HistoryUpdater <: POMDPs.Updater
    m::PostBiddingProblem
end

updater(p::GreedyPolicy) = HistoryUpdater(p.m)

function action(p::GreedyPolicy, b::ObservationDistribution)
    if b.o == nothing
        return Action(n=1)
    end
    return GreedyAction(p.m, b.o.p1, b.o.p1_d, b.o.p2_d, b.o.p3_d, b.o.p4_d, 1, b.o.t)
end

function extract_belief(up::HistoryUpdater, ::BasicPOMCP.POMCPObsNode{Action,Observation})
    return ObservationDistribution(up.m, )
end

initialize_belief(up::HistoryUpdater, d::Any) = d

function POMDPs.update(up::HistoryUpdater, b, a, o)
    return ObservationDistribution(b.m, o)
end

function BasicPOMCP.extract_belief(up::HistoryUpdater, node::BeliefNode)
    if node.node==1 && !isdefined(node.tree.o_labels, node.node)
        missing
    else
        o = node.tree.o_labels[node.node]
        return ObservationDistribution(up.m, o)
    end
end

Random.seed!(1234)
rng = MersenneTwister(12)
contract = Contract(suite=diamonds, n=8, p=1)
pomdp = PostBiddingProblem(contract=contract, n_cards=13)
solver = POMCPSolver(tree_queries=10000, estimate_value=RolloutEstimator(GreedyPolicy(pomdp)), rng=rng)
planner = solve(solver, pomdp)

greedy_policy = GreedyPolicy(pomdp)

random_policy = RandomPolicy(pomdp)

# b = initialstate_distribution(pomdp)
# a = action(planner, b)
# println(a)

b_updater = HistoryUpdater(pomdp)

rsum = 0.0
for i in 1:20
    println("Deal $(i)")
    total = 0.0
    for (s,a,r,sp,o) in stepthrough(pomdp, planner, b_updater, "s,a,r,sp,o")
        @show (s,a,r,sp,o)
        global rsum += r
        total += r
    end
    println("Deal $(i) reward: $(total)")
    println("Avg reward: $(rsum / i)")
end
println("Undiscounted POMCP reward was $rsum.")

# rsum = 0.0
# for i in 1:20
#     println("Deal $(i)")
#     total = 0.0
#     for (s,a,r,sp,o) in stepthrough(pomdp, greedy_policy, b_updater, "s,a,r,sp,o")
#         @show (s,a,r,sp,o)
#         global rsum += r
#         total += r
#     end
#     println("Deal $(i) reward: $(total)")
#     println("Avg reward: $(rsum / i)")
# end
# println("Undiscounted greedy reward was $rsum.")

# rsum = 0.0
# for i in 1:20
#     println("Deal $(i)")
#     total = 0.0
#     for (s,a,r,sp,o) in stepthrough(pomdp, random_policy, b_updater, "s,a,r,sp,o")
#         @show (s,a,r,sp,o)
#         global rsum += r
#         total += r
#     end
#     println("Deal $(i) reward: $(total)")
#     println("Avg reward: $(rsum / i)")
# end
# println("Undiscounted random reward was $rsum.")
