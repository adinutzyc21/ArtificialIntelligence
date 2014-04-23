import util

def newVal(mdp,state,gamma,oldV):
    actions = mdp.getPossibleActions(state)
    sums = util.Counter()
    action = None
    for action in actions:
        sums[action] = qVal(mdp,state,action,gamma,oldV)
    maxV = sums[action]
    #for a in sums:
    #    maxV = sums[a]
    for a in sums:
        if sums[a] > maxV:
            maxV = sums[a]
    return maxV

def qVal(mdp,state,action,gamma,oldV):
    sum = 0.0
    for ns_p in mdp.getTransitionStatesAndProbs(state, action):
        nextState = ns_p[0]
        P = ns_p[1]
        R = mdp.getReward(state, action, nextState)
        sum += P * (R + gamma * oldV[nextState])
    return sum