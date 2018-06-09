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
        self.values = util.Counter() # A Counter is a dict with default 0
        
        if self.iterations != 0:
            self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        temp = util.Counter()
        for k in range(self.iterations):
            for state in self.mdp.getStates():
                actionValues = []
                for action in self.mdp.getPossibleActions(state):
                    actionValues.append(self.computeQValueFromValues(state,action))
                if len( actionValues ) != 0 :
                    temp[state] = max(actionValues)
            for state in temp:
                self.values[state] = temp[state] 

                


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
        q = 0
        for transprob in self.mdp.getTransitionStatesAndProbs(state, action):
            trans, prob = transprob
            q = q + prob*( self.mdp.getReward(state, action, trans) + self.discount * self.values[trans] )
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestQ = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state,action)
            if q > bestQ:
                bestQ = q
                bestAction = action
        return bestAction

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
        counter = 0

        for i in range(self.iterations):
          for state in self.mdp.getStates():
            q = util.Counter()
            max = float("-inf")
            for action in self.mdp.getPossibleActions(state):
                val = self.computeQValueFromValues(state, action)
                if val > max:
                    max = val
                q[action] = val
            
            self.values[state] = q[q.argMax()]
            
            counter += 1
            if counter >= self.iterations:
              return

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
        predecessors = {}
        for state in self.mdp.getStates():
          predecessors[state] = set()
        priority = util.PriorityQueue()
        for state in self.mdp.getStates():
          q = util.Counter()
          for action in self.mdp.getPossibleActions(state):
            T = self.mdp.getTransitionStatesAndProbs(state, action)
            for transprob in self.mdp.getTransitionStatesAndProbs(state, action):
                nextState, prob = transprob
                if prob != 0:
                    predecessors[nextState].add(state)
            q[action] = self.computeQValueFromValues(state, action)
          if not self.mdp.isTerminal(state): 
            priority.update(state, -1*abs(self.values[state] - q[q.argMax()]))
        for i in xrange(self.iterations):
          if priority.isEmpty():
            return
          state = priority.pop()
          if self.mdp.isTerminal(state) == False:
            q = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              q[action] = self.computeQValueFromValues(state, action)
            self.values[state] = q[q.argMax()]
          for p in predecessors[state]:
            q_pred = util.Counter()
            for action in self.mdp.getPossibleActions(p):
              q_pred[action] = self.computeQValueFromValues(p, action)
            if abs(self.values[p] - q_pred[q_pred.argMax()]) > self.theta:
              priority.update(p, -1*abs(self.values[p] - q_pred[q_pred.argMax()]))
