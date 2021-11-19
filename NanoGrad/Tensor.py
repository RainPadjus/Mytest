import numpy as np


###### ######  #   #  #####  ####  ###
   #   ##      ##  #  #      #  #  #  #
   #   ######  # # #  #####  #  #  ###
   #   ##      #  ##      #  #  #  #  #
   #   ######  #   #  #####  ####  #  #


class Tensor (object):

    def __init__(self, data,
                 autograd=False, 
                 creators=None, 
                 creation_op = None,
                 id =None):


        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        
        self.children = {}
        if(id is None):
            id = np.random.randint(0, 100000)
        self.id = id

        if (creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] +=1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if(cnt!=0):
                return False
        return True


    def backward(self, grad=None, grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("cant back more than 1")
                else:
                    self.children[grad_origin.id] -=1

            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad

            if(self.creators is not None and
                    (self.all_children_grads_accounted_for() or
                        grad_origin is None)):
                        
                        if(self.creation_op == "add"):
                            self.creators[0].backward(self.grad, self)
                            self.creators[1].backward(self.grad, self)
                        
                        if(self.creation_op =="neg"):
                            self.creators[0].backward(self.grad.__neg__())
                        
                        if(self.creation_op =="sub"):
                            new = Tensor(self.grad.data)
                            self.creators[0].backward(new, self)
                            new = Tensor(self.grad.__neg__().data)
                            self.creators[1].backward(new, self)
                        
                        if(self.creation_op == "mul"):
                            new= self.grad * self.creators[1]
                            self.creators[0].backward(new, self)
                            new= self.grad * self.creators[0]
                            self.creators[1].backward(new, self)

                        if(self.creation_op=="mm"):
                            act = self.creators[0]
                            weights = self.creators[1]
                            new= self.grad.mm(weights.transpose())
                            act.backward(new)
                            new = self.grad.transpose().mm(act).transpose()
                            weights.backward(new)

                        if(self.creation_op == "transpose"):
                            self.creators[0].backward(self.grad.transpose())

                        if("sum" in self.creation_op):
                            dim = int(self.creation_op.split("_")[1])
                            ds = self.creators[0].data.shape[dim]
                            self.creators[0].backward(self.grad.expand(dim, ds))
                        
                        if("expand" in self.creation_op):
                            dim = int(self.creation_op.split("_")[1])
                            self.creators[0].backward(self.grad.sum(dim))


    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data *-1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data *-1)


    def __add__(self, other):
        if(self.autograd and other.autograd):

            return Tensor(self.data + other.data,
                          autograd=True,
                          creators = [self, other],
                          creation_op = "add")

        return Tensor(self.data + other.data)
    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators = [self, other],
                          creation_op = "sub")
        
            return Tensor(self.data-other.data)
    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd = True,
                          creators = [self, other],
                          creation_op = "mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd =True,
                          creators=[self],
                          creation_op = "sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data= new_data.transpose(trans_cmd)

        if(self.autograd):
            return Tensor(new_data,
                          autograd = True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd = True,
                          creators=[self],
                          creation_op = "transpose")
        
        return Tensor(self.data.transpose())


    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op ="mm")

        return Tensor(self.data.dot(x.data))

    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())



##### ####  ####
#     #     #   #
##### # ### #   #
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

class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)
        


np.random.seed(0)

data = Tensor(np.array([[0,0], [0,1], [1,0], [1,1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd = True)

model = Sequential([Linear(2,3), Linear(3,1)])
criterion = MSELoss()

optim = SGD(parameters = model.get_parameters(), alpha=0.05)

for i in range(10):
    pred = model.forward(data)

    loss = criterion.forward(pred, target)
    
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print("STEP {} \ LOSS {}".format(i, loss))
