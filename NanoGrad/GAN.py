import numpy as np


###### ######  #   #  #####  ####  ###
   #   ##      ##  #  #      #  #  #  #
   #   ######  # # #  #####  #  #  ###
   #   ##      #  ##      #  #  #  #  #
   #   ######  #   #  #####  ####  #  #
class Tensor (object):
    
    def __init__(self,data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        if(id is None):
            self.id = np.random.randint(0,100000)
        else:
            self.id = id
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        
        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True 
        
    def backward(self,grad=None, grad_origin=None):
        if(self.autograd):
 
            if(grad is None):
                grad = Tensor(np.ones_like(self.data))

            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("OP!")
                else:
                    self.children[grad_origin.id] -= 1

            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad
            
            assert grad.autograd == False
            
            if(self.creators is not None and 
               (self.all_children_grads_accounted_for() or 
                grad_origin is None)):

                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if(self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if(self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)                    
                    
                if(self.creation_op == "mm"):
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
                    
                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                    
                if(self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())
                    
                if(self.creation_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                
                if(self.creation_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))
                    
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        return Tensor(self.data * other.data)    

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    def sigmoid(self):
        if(self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if(self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))
        
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())  

##### ####  #### 
#     #     #   #
##### #  ## #   #
    # #   # #   #
##### ##### # ##


class SGD(object):

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *=0
    
    def step(self, zero=True):

        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if(zero):
                p.grad.data *= 0



#        #   #   # ####  ###
#        #   #   # #     #  #
#       # #   ###  ####  # #
#      #####   #   #     #  #
####  #     #  #   ####  #   #

class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = (np.random.randn(n_inputs, n_outputs)*np.sqrt(2.0/n_inputs))
        self.weight =Tensor(W, autograd = True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight)+self.bias.expand(0, len(input.data))


class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params +=l.get_parameters()
        return params

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()

class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)


from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



#####GAN
np.random.seed(0)

#x_train = x_train.astype("float32") / 255

#x_train = x_train.reshape(60000, 28*28)[:1000]
#y_train = y_train[:1000]


#x_train_noise = np.random.random(x_train.shape)

#one = np.ones(y_train.shape)
#zero = np.zeros(y_train.shape)

#x_train = np.concatenate((x_train, x_train_noise))
#y_train = np.concatenate((one, zero))



###############

#np.random.seed(0)

#x_train = x_train.astype("float32") / 255

#x_train = x_train.reshape(60000, 28*28)[:1000]
#y_train = y_train[:1000]

#x_train_nose = np.random.random(x_train.shape)



#new = [[0 for i in range(10)] for j in range(len(y_train))]

#for x, i in enumerate(y_train):
#    new[x][i] = 1

#y_train = np.array(new)
#x_train = np.array(x_train)

#data = Tensor(x_train, autograd = True)
#target = Tensor(y_train, autograd = True)

#print(data.data)
#print(target.data)

data_disc = Tensor(np.array([[1,1,1,0,0,0,0,0,0,0], 
                        [1,1,1,0,0,0,0,0,0,0], 
                        [1,1,1,0,0,0,0,0,0,0], 
                        [1,1,0,0,0,0,0,0,0,0], 
                        [0,1,0,1,1,0,1,0,1,0], 
                        [0,1,0,1,0,0,1,0,1,1], 
                        [0,1,1,1,1,0,0,1,0,0], 
                        [0,0,0,0,1,0,0,0,1,0]]), autograd=True)

target_disc = Tensor(np.array([[1], [1], [1], [1], [0], [0], [0], [0]]), autograd = True)



#model = Sequential([Linear(28*28,40), Tanh(), Linear(40, 10), Tanh(), Linear(10,10)])
#model = Sequential([Linear(28*28,60), Tanh(), Linear(60,20), Tanh(), Linear(20,10), Sigmoid()])

generator = Sequential([Linear(1, 30), Sigmoid(), Linear(30,10), Sigmoid(), Tanh(), Linear(10, 10), Sigmoid()])
discriminator = Sequential([Linear(10,20), Tanh(), Linear(20, 10), Sigmoid(), Linear(10,1), Sigmoid()])

criterion = MSELoss()

optim_discriminator = SGD(parameters = discriminator.get_parameters(), alpha=0.005)
optim_generator= SGD(parameters = generator.get_parameters(), alpha=0.005)

target_gen = Tensor(np.ones((10,1)), autograd = True)

for epoch in range(100):
    #Training Discriminator
    for i in range(1000):
        pred = discriminator.forward(data_disc)
        loss = criterion.forward(pred, target_disc)
        if i%1000==0:
            print("DISCRIMINATOR : LOSS {}".format(np.sum(loss.data)))

        loss.backward(Tensor(np.ones_like(loss.data)))
        optim_discriminator.step()
        np.set_printoptions(suppress=True)




    

    #Training Generator
    for i in range(1000):
        data_gen = Tensor(np.random.random((10,1)), autograd =True)
        pred = generator.forward(data_gen)
        #print("TARGET:", target)
        #d_test = Tensor(np.array(discriminator.forward(pred).data.copy()))
        d_test = discriminator.forward(pred)
        #print("PRED: ", pred)
        #print("Dtest: ,",d_test)
        loss = criterion.forward(d_test, target_gen)
        if i%1000==0:
            print("Generator :  LOSS {}".format(np.sum(loss.data)))

        loss.backward(Tensor(np.ones_like(loss.data)))
        optim_generator.step()

    print("EPOCH: ", epoch)
    print("Generating data: ")
    print(generator.forward(Tensor(np.random.random(1))))




#for z in [1,22,33,55,100,10, 66,67,68,69,70]:
#    x= x_train[z]
#    x = Tensor(x)
#    pp = model.forward(x)
#    print("PRED: {} / REAL: {}".format(pp.data[z].argmax(), y_train[z].argmax()))
