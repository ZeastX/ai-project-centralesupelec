import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        self.learning_rate = 0.05
        self.numTrainingGames = 44000
        self.batch_size = 100
        self.parameters = [nn.Parameter(self.state_size,125),nn.Parameter(1,125),nn.Parameter(125,50),nn.Parameter(1,50),nn.Parameter(50,self.num_actions),nn.Parameter(1,self.num_actions)]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        loss = nn.SquareLoss(self.run(states),Q_target)
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        relu0 = nn.ReLU(nn.AddBias(nn.Linear(states,self.parameters[0]),self.parameters[1]))
        relu1 = nn.ReLU(nn.AddBias(nn.Linear(relu0,self.parameters[2]),self.parameters[3]))
        res = nn.AddBias(nn.Linear(relu1,self.parameters[4]),self.parameters[5])
        return res

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        loss = self.get_loss(states,Q_target)
        grad = nn.gradients(loss,self.parameters)
        for i in range(len(self.parameters)):    
            self.parameters[i].update(grad[i],-self.learning_rate)
        return None
