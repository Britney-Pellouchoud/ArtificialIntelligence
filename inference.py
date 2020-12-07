 # inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        z = self.total()
        if z:
            for key in self.keys():
                val = self.get(key)
                self.update({key: val/z})

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        self.normalize()
        #print('called')
        item = random.choices(list(self.keys()), self.values())
        return item[0]

        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        if ghostPosition == jailPosition:
            if noisyDistance == None:
                return 1.0
            else:
                return 0.0
        if noisyDistance == None and ghostPosition != jailPosition:
            return 0.0
        prob_noisy_given_true = busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))
        return prob_noisy_given_true
        #raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """

        for position in self.allPositions:
            obsProb = self.getObservationProb(observation, gameState.getPacmanPosition(), position, self.getJailPosition())
            self.beliefs[position] *= obsProb




        self.beliefs.normalize()

        '''
        obs = 0
        for pos in self.allPositions:
            obs_given_ghost_at_i = self.getObservationProb(observation, gameState.getPacmanPosition(), pos, self.getJailPosition())
            ghost_at_i = self.beliefs[pos]
            obs = sum([self.getObservationProb(observation, gameState.getPacmanPosition(), i, self.getJailPosition()) for i in self.allPositions])
            if obs:
                self.beliefs[pos] = (obs_given_ghost_at_i*ghost_at_i)/obs
            else:
                self.beliefs[pos] = 0.0 #This passes autograder tests, but not sure why. May run into bugs later because of this line.
        '''

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        beliefs = self.beliefs.copy()
        for i in beliefs.keys():
            beliefs[i] = 0

        for prevPos in self.allPositions:
            newPosDist = self.getPositionDistribution(gameState, prevPos)
            for newPos in self.allPositions:
                beliefs[newPos] +=  newPosDist[newPos]*self.beliefs[prevPos]
        self.beliefs = beliefs
        self.beliefs.normalize()

        '''
        distances = DiscreteDistribution() #create a new dictionary
        for origPos in self.allPositions:
            newDist = self.getPositionDistribution(gameState, origPos) #get all the new distances
            origProb = self.beliefs[origPos]
            for newPos in newDist.keys():       
                distances[newPos] += origProb * newDist[newPos] #bayes nets inference
        self.beliefs = distances
        '''

        '''
        dist_dict = {}
        for pos in self.allPositions:
            dist_dict[pos] = (self.getPositionDistribution(gameState, pos), pos)
        print(str(dist_dict))
        #dist_list has all position distributions
        for p in self.allPositions:
            self.beliefs[p] = sum([dist_dict[position][0][p]*self.beliefs[dist_dict[position][1]] for position in self.allPositions])'''




    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        numPositions = len(self.legalPositions)
        
        perPos = self.numParticles/numPositions

        for position in self.legalPositions:
            i = 0
            while i < perPos:
                self.particles.append(position)
                i += 1
        "*** YOUR CODE HERE ***"
        #raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        current_beliefs = self.getBeliefDistribution()
        new_beliefs = current_beliefs.copy()
        for particle in current_beliefs.keys():
            weight = self.getObservationProb(observation, gameState.getPacmanPosition(), particle, self.getJailPosition())
            new_beliefs[particle] = new_beliefs[particle]*weight

        #assuming this is right,
        if new_beliefs.total() == 0:
            self.initializeUniformly(gameState)
            return

        newParticles = []
        i = 0
        while i < self.numParticles:
            newParticles.append(new_beliefs.sample())
            i += 1
        self.particles = newParticles

        # beliefs = self.getBeliefDistribution()
        # pacmanPosition = gameState.getPacmanPosition()
        # newBeliefs = DiscreteDistribution()
        # for particle in self.particles:
        #     a = self.getObservationProb(observation, pacmanPosition, particle, self.getJailPosition())
        #     originalAndNew = a * beliefs[particle]
        #     newBeliefs[particle] = originalAndNew
        # if not len(newBeliefs):
        #     self.initializeUniformly(gameState)
        #     return
        # newBeliefs.normalize()
        # newParticles = []
        # i = 0
        # while i < self.numParticles:
        #     newParticles.append(newBeliefs.sample())
        #     i += 1
        # self.particles = newParticles




        #raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        new_particles = []
        for particle in self.particles:
            new_dist = self.getPositionDistribution(gameState, particle)
            new_particles.append(new_dist.sample())
        self.particles = new_particles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        particles = {}
        for p in self.particles:
            particles[p] = self.particles.count(p)
        particleDist = DiscreteDistribution(particles)
        #print(self.numParticles)
        #print(particleDist)
        particleDist.normalize()
        #print(particleDist)
        return particleDist


        #raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        # self.particles = []
        # numPositions = len(self.legalPositions)
        #
        # perPos = self.numParticles / numPositions
        #
        # for position in self.legalPositions:
        #     i = 0
        #     while i < perPos:
        #         self.particles.append(position)
        #         i += 1
        self.particles = []
        legal_positions = self.legalPositions
        all_comb_positions = list(itertools.product(legal_positions, repeat=self.numGhosts))
        #print(all_comb_positions)
        # perPos = self.numParticles/len(all_comb_positions)
        # for position in all_comb_positions:
        #     i = 0
        #     while i < perPos:
        #         self.particles.append(position)
        #         i+=1
        # #print(self.particles)
        # random.shuffle(self.particles)
        random.shuffle(all_comb_positions)
        self.particles = all_comb_positions
        "*** YOUR CODE HERE ***"

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # current_beliefs = self.getBeliefDistribution()
        # new_beliefs = current_beliefs.copy()
        # for particle in current_beliefs.keys():
        #     weight = self.getObservationProb(observation, gameState.getPacmanPosition(), particle,
        #                                      self.getJailPosition())
        #     new_beliefs[particle] = new_beliefs[particle] * weight
        #
        # # assuming this is right,
        # if new_beliefs.total() == 0:
        #     self.initializeUniformly(gameState)
        #     return
        #
        # newParticles = []
        # i = 0
        # while i < self.numParticles:
        #     newParticles.append(new_beliefs.sample())
        #     i += 1
        # self.particles = newParticles
        #print(observation)

        # print("belief distribution", self.getBeliefDistribution())
        # all_new_particles = []
        # for i in range(self.numGhosts):
        #     current_beliefs = self.getBeliefDistribution()
        #     new_beliefs = current_beliefs.copy()
        #     for particle in current_beliefs.keys():
        #         #print(particle[i])
        #         weight = self.getObservationProb(observation[i], gameState.getPacmanPosition(), particle[i],
        #                                          self.getJailPosition(i))
        #         new_beliefs[particle] = new_beliefs[particle] * weight
        #
        #     # assuming this is right,
        #     if new_beliefs.total() == 0:
        #         self.initializeUniformly(gameState)
        #         return
        #
        #     i = 0
        #     while i < self.numParticles:
        #         all_new_particles.append(new_beliefs.sample())
        #         i += 1
        particle_dist = self.getBeliefDistribution()
        for particle in particle_dist.keys():
            observation_prob_list = []
            for i in range(self.numGhosts):
                observation_prob_list.append(self.getObservationProb(observation[i], gameState.getPacmanPosition(), particle[i], self.getJailPosition(i)))
            u = 1
            for elt in observation_prob_list:
                u *= elt
            particle_dist[particle] = particle_dist[particle]*u

        if particle_dist.total() == 0:
            self.initializeUniformly(gameState)
            return

        new_particles = []
        i = 0
        while i < self.numParticles:
            new_particles.append(particle_dist.sample())
            i += 1
        self.particles = new_particles
        #raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            newPositions = []
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
