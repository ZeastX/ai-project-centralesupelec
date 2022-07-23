import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x,self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        dotProduct = self.run(x)
        scal = nn.as_scalar(dotProduct)
        if scal >= 0:
            return 1
        else :
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        finished = False
        L=nn.np.zeros(dataset.x.data.shape[0])
        counter = 0
        for x, y in dataset.iterate_forever(1):
            if counter == L.shape[0]:
                counter = 0
            if nn.as_scalar(y) - self.get_prediction(x) == 0:
                L[counter] = 1
            else :
                L[counter] = 0
            direction = nn.Constant((nn.as_scalar(y) - self.get_prediction(x) ) * x.data)
            multiplier = 0.5
            self.w.update(direction,multiplier)
            counter += 1
            percentage = L.sum()/L.shape[0]
            print("Progression : " + str(percentage*100) + "%")
            if [x for x in L] == [1 for i in range(L.shape[0])]:
                finished = True
            if finished:
                break
        print("Finished training")
        return None


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.m1 = nn.Parameter(1,60)
        self.m2 = nn.Parameter(60,25)
        self.b1 = nn.Parameter(1,60)
        self.b2 = nn.Parameter(1,25)
        self.m3 = nn.Parameter(25,1)
        self.b3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        relu = nn.ReLU(nn.AddBias(nn.Linear(x,self.m1),self.b1))
        reluBias = relu
        reluBias2=nn.ReLU(nn.AddBias(nn.Linear(reluBias,self.m2),self.b2))
        y_predict = nn.AddBias(nn.Linear(reluBias2,self.m3),self.b3)
        return y_predict

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        y_pred = self.run(x)
        loss = nn.SquareLoss(y_pred,y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        l_rate=-0.01
        for x,y in dataset.iterate_forever(20):
            loss = self.get_loss(x,y)
            g_m1,g_m2,g_b1,g_b2,g_m3,g_b3 = nn.gradients(loss,[self.m1,self.m2,self.b1,self.b2,self.m3,self.b3])
            self.m1.update(g_m1,l_rate)
            self.m2.update(g_m2,l_rate)
            self.b1.update(g_b1,l_rate)
            self.b2.update(g_b2,l_rate)
            self.m3.update(g_m3,l_rate)
            self.b3.update(g_b3,l_rate)
            for x1,y1 in dataset.iterate_once(dataset.x.data.shape[0]):
                absolute_loss = self.get_loss(x1,y1)
            print(nn.as_scalar(absolute_loss))
            if nn.as_scalar(absolute_loss) < 0.02:
                break
        return None

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.layer_size = 150
        self.m1 = nn.Parameter(784,self.layer_size)
        self.b1 = nn.Parameter(1,self.layer_size)
        self.m2 = nn.Parameter(self.layer_size,self.layer_size//2)
        self.b2 = nn.Parameter(1,self.layer_size//2)
        self.m3 = nn.Parameter(self.layer_size//2,self.layer_size//4)
        self.b3 = nn.Parameter(1,self.layer_size//4)

        self.m4 = nn.Parameter(self.layer_size//4,10)
        self.b4 = nn.Parameter(1,10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        relu = nn.ReLU(nn.AddBias(nn.Linear(x,self.m1),self.b1))
        relu2 = nn.ReLU(nn.AddBias(nn.Linear(relu,self.m2),self.b2))
        relu3 = nn.ReLU(nn.AddBias(nn.Linear(relu2,self.m3),self.b3))
        y_pred = nn.AddBias(nn.Linear(relu3,self.m4),self.b4)
        return y_pred

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        l_rate=-0.13
        for x,y in dataset.iterate_forever(200):
            loss = self.get_loss(x,y)
            g_m1,g_m2,g_b1,g_b2,g_m3,g_b3,g_m4,g_b4 = nn.gradients(loss,[self.m1,self.m2,self.b1,self.b2,self.m3,self.b3,self.m4,self.b4])
            self.m1.update(g_m1,l_rate)
            self.m2.update(g_m2,l_rate)
            self.b1.update(g_b1,l_rate)
            self.b2.update(g_b2,l_rate)
            self.m3.update(g_m3,l_rate)
            self.b3.update(g_b3,l_rate)
            self.m4.update(g_m4,l_rate)
            self.b4.update(g_b4,l_rate)
            print(dataset.get_validation_accuracy())
            if dataset.get_validation_accuracy() > 0.975:
                break
        return None

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.m1 = nn.Parameter(47,47)
        self.b1= nn.Parameter(1,47)
        self.m_hid=nn.Parameter(47,47)
        self.m2=nn.Parameter(47,50)
        self.b2=nn.Parameter(1,50)
        self.m3=nn.Parameter(50,25)
        self.b3=nn.Parameter(1,25)
        self.m4=nn.Parameter(25,5)
        self.b4=nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        res = nn.AddBias(nn.Linear(xs[0],self.m1),self.b1)
        for i in range(1,len(xs)):
            res = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(xs[i],self.m1),self.b1),nn.Linear(res,self.m_hid)))
        res = nn.ReLU(nn.AddBias(nn.Linear(res,self.m2),self.b2))
        res1 = nn.ReLU(nn.AddBias(nn.Linear(res,self.m3),self.b3))
        y_pred = nn.AddBias(nn.Linear(res1,self.m4),self.b4)
        return y_pred

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        loss = nn.SoftmaxLoss(self.run(xs),y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        l_rate=-0.07
        for x,y in dataset.iterate_forever(300):
            loss = self.get_loss(x,y)
            g_m1,g_m2,g_b1,g_b2,g_mhid,g_m3,g_b3,g_m4,g_b4 = nn.gradients(loss,[self.m1,self.m2,self.b1,self.b2,self.m_hid,self.m3,self.b3,self.m4,self.b4])
            self.m1.update(g_m1,l_rate)
            self.m2.update(g_m2,l_rate)
            self.b1.update(g_b1,l_rate)
            self.b2.update(g_b2,l_rate)
            self.m_hid.update(g_mhid,l_rate)
            self.m3.update(g_m3,l_rate)
            self.b3.update(g_b3,l_rate)
            self.m4.update(g_m4,l_rate)
            self.b4.update(g_b4,l_rate)

            if dataset.get_validation_accuracy() > 0.88:
                break
        return None
