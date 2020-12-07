# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0 -- stores values of all states in mdp
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        all_states = self.mdp.getStates()
        for _ in range(self.iterations):
            next_values = self.values.copy()
            for state in all_states:
                actions = self.mdp.getPossibleActions(state)
                if actions:
                    q_values = [self.computeQValueFromValues(state, action) for action in actions]
                    max_q_val = max(q_values)
                    next_values[state] = max_q_val
                    #print('HERE', self.values)
            self.values = next_values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) =∑s′T(s, a, s
        # ')[R(s,a,s')+γV(s
        # ')]
        states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action) #[(next state, prob), (next state, prob)]
        Q_value = 0
        for next_state, prob in states_and_probs:
            T = prob
            reward = self.mdp.getReward(state, action, next_state)
            value = self.getValue(next_state)
            Q_value += T*(reward + self.discount*value)
        return Q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        q_list = []
        if not actions:
            return None
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            q_list.append((q_value, action))
        best_action = max(q_list, key = lambda val: val[0])[1]
        return best_action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        for num in range(self.iterations):
            state = all_states[num%len(all_states)]
            actions = self.mdp.getPossibleActions(state)
            if actions:
                q_values = [self.computeQValueFromValues(state, action) for action in actions]
                max_q_val = max(q_values)
                self.values[state] = max_q_val


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()

        priority_queue = util.PriorityQueue()
        for state in all_states:
            if not self.mdp.isTerminal(state):
                max_q_val = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - max_q_val)
                priority_queue.push(state, -diff)

        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                return
            s = priority_queue.pop()
            if not self.mdp.isTerminal(s):
                q_values = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
                max_q_val = max(q_values)
                self.values[s] = max_q_val

            predecessors = set()
            for state in all_states:
                #if state != s:
                possible_actions = self.mdp.getPossibleActions(state)
                for action in possible_actions:
                    states_with_this_action = [item[0] for item in self.mdp.getTransitionStatesAndProbs(state, action)]
                    if s in states_with_this_action:
                        predecessors.add(state)

            for p in predecessors:
                max_q_val = max(
                    [self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)])
                diff = abs(self.values[p] - max_q_val)

                if diff > self.theta:
                    priority_queue.update(p, -diff)





