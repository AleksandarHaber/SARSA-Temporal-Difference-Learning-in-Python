# -*- coding: utf-8 -*-
"""
- Temporal Difference Learning and Control Tutorial 
- Reinforcement Learning Tutorial
- On-Policy SARSA (State-Action-Reward-State-Action) Reinforcement Learning 

Author: Aleksandar Haber
Date: January 2023

This Python file contains a class definition that implements 
the SARSA Temporal Difference Learning Algorithm

"""
import numpy as np


class SARSA_Learning:
    
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS: 
    # env - Frozen Lake environment
    # alpha - step size 
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes
    
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes):
     
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.stateNumber=env.observation_space.n
        self.actionNumber=env.action_space.n 
        self.numberEpisodes=numberEpisodes
        # this vector is the learned policy
        self.learnedPolicy=np.zeros(env.observation_space.n)
        # this matrix is the action value function matrix 
        # its entries are (s,a), where s is the state number and action is the action number
        # s=0,1,2,\ldots,15, a=0,1,2,3
        self.Qmatrix=np.zeros((self.stateNumber,self.actionNumber))
        
    
    ###########################################################################
    #   END - __init__ function
    ###########################################################################
    
    ###########################################################################
    #    START - function for selecting an action: epsilon-greedy approach
    ###########################################################################
    # this function selects an action on the basis of the current state 
    # INPUTS: 
    # state - state for which to compute the action
    # index - index of the current episode
    def selectAction(self,state,index):
        
        # first 100 episodes we select completely random actions to avoid being stuck
        if index<100:
            return np.random.choice(self.actionNumber)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        randomNumber=np.random.random()
          
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            self.epsilon=0.9*self.epsilon
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)            
        
        # otherwise, we are selecting greedy actions
        else:
            self.epsilon=0.9*self.epsilon
            # we return the index where actionValueMatrixEstimate[state,:] has the max value
            return np.random.choice(np.where(self.Qmatrix[state,:]==np.max(self.Qmatrix[state,:]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example 
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple
    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################
    
    
    ###########################################################################
    #    START - function for simulating an episode
    ###########################################################################
     
    def simulateEpisodes(self):
        
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            
            # reset the environment at the beginning of every episode
            (stateS,prob)=self.env.reset()
            
            # select an action on the basis of the initial state
            actionA = self.selectAction(stateS,indexEpisode)
            
            print("Simulating episode {}".format(indexEpisode))
            
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            while not terminalState:
                         
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (stateSprime, rewardPrime, terminalState,_,_) = self.env.step(actionA)          
                
                # next action
                actionAprime = self.selectAction(stateSprime,indexEpisode)
                
                if not terminalState:
                    error=rewardPrime+self.gamma*self.Qmatrix[stateSprime,actionAprime]-self.Qmatrix[stateS,actionA]
                    self.Qmatrix[stateS,actionA]=self.Qmatrix[stateS,actionA]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error=rewardPrime-self.Qmatrix[stateS,actionA]
                    self.Qmatrix[stateS,actionA]=self.Qmatrix[stateS,actionA]+self.alpha*error
                                    
                stateS=stateSprime
                actionA=actionAprime
    
    ###########################################################################
    #    END - function for simulating an episode
    ###########################################################################
                 
    
    ###########################################################################
    #    START - function for computing the final policy
    ###########################################################################            
    def computeFinalPolicy(self):
        
        # now we compute the final learned policy
        for indexS in range(self.stateNumber):
            # we use np.random.choice() because in theory, we might have several identical maximums
            self.learnedPolicy[indexS]=np.random.choice(np.where(self.Qmatrix[indexS]==np.max(self.Qmatrix[indexS]))[0])
    
    ###########################################################################
    #    END - function for computing the final policy
    ###########################################################################            
                
                
                
                
                
                
                
                
                
                
                
                
                
            
            
            
            
        
        
        
        
        
        