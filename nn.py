import math
import random

#activations: each can return deriv or activ
def linear(x,deriv=False):
    if not deriv:
        return x
    else: return 1
def sigmoid(x,deriv=False):
    sig=1/(1+math.exp(-x))
    if not deriv:
        return sig
    else: return sig*(1-sig)
def relu(x,deriv=False):
    if not deriv:
        return max(0,x)
    else: return (x>0)*1
def mse(y_pred,y_true,loss=False):
    if len(y_pred)!=len(y_true):raise ValueError("MSE:input sizes don't match!")
    if not loss:
        result=[0.0]*len(y_pred)
        for i in range(len(y_pred)):
            result[i]=y_pred[i]-y_true[i]
    else:
        return sum(mse(y_pred, y_true))**2
    return result
class Model:
    class InputLayer:
        def __init__(self,layer_size):
            self.layer_size=layer_size
            self.activation_value=[0.0]*layer_size
    class Layer:
        weights=[]
        bias=[]
        weight_grads=[]
        bias_grads=[]
        @staticmethod
        def _matmul(a, b):
            result=[[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
            for i in range(len(a)):
                for j in range(len(b[0])):
                    for k in range(len(b)):
                        result[i][j]+=a[i][k]*b[k][j]
            return result
        @staticmethod
        def _scalarmul(a, b):
            result=[0.0]*len(a)
            for i in range(len(a)):
                result[i]=a[i]*b
            return result
        @staticmethod
        def _elemmul(a, b):
            result=[0.0]*len(a)
            for i in range(len(a)):
                result[i]=a[i]*b[i]
            return result
        @staticmethod
        def _matrixconv(x, rot=0):
            result=[]
            if isinstance(x[0],list):
                for i in range(len(x)):
                    for j in range(len(x[0])):
                       result.append(x[i][j])
            else:
                if rot==0:
                    for i in range(len(x)):
                        result.append([x[i]])
                if rot==1:
                    result.append(x)
            return result
        def __init__(self,layer_size,activation):
            self.layer_size=layer_size
            self.value=[0.0]*layer_size
            self.activation_value=[0.0]*layer_size
            self.activation=activation
        def forw(self,x):
            for i in range(len(self.weights[0])):
                self.value[i]=self.bias[i]
                for j in range(len(self.weights)):
                    self.value[i]+=x[j]*self.weights[j][i]
                self.activation_value[i]=self.activation(self.value[i])
        def back(self,x,delta):
            for i in range(self.layer_size):
                delta[i]*=self.activation(self.value[i],True)
            for i in range(len(delta)):
                self.bias_grads[i]+=delta[i]
            gradients=self._matmul(self._matrixconv(x, 0), self._matrixconv(delta, 1))
            for i in range(len(x)):
                for j in range(self.layer_size):
                    self.weight_grads[i][j]+=gradients[i][j]
            return self._matrixconv(self._matmul(self.weights, self._matrixconv(delta, 0)))
    @staticmethod
    def _he_initializer(in_size, buffer):
        buffer+=1 #just muffles an error, this is unused.
        return random.gauss(0, math.sqrt(2.0 / in_size))
    @staticmethod
    def _xavier_initializer(in_size, out_size):
        limit = math.sqrt(6.0 / (in_size + out_size))
        return random.uniform(-limit, limit)
    def __init__(self,layers):
        self.layers=layers
        points=[-.5,.5]
        for i in range(1,len(self.layers)):
            func=self._xavier_initializer
            if self.layers[i].activation(points[0])==relu(points[0]) and self.layers[i].activation(points[1])==relu(points[1]):
                func=self._he_initializer
            in_size=self.layers[i-1].layer_size
            out_size=self.layers[i].layer_size
            self.layers[i].weights=[[func(in_size, out_size) for _ in range(out_size)] for i in range(in_size)]
            self.layers[i].weight_grads=[[0.0 for _ in range(out_size)] for i in range(in_size)]
            self.layers[i].bias=[func(in_size, out_size) for i in range(out_size)]
            self.layers[i].bias_grads=[0.0 for i in range(out_size)]
    def inference(self,x):
        if len(x)!=self.layers[0].layer_size: raise ValueError("Model:Forward:input size doesn't match!")
        self.layers[0].activation_value=x
        for i in range(1,len(self.layers)):
            self.layers[i].forw(self.layers[i-1].activation_value)
        return self.layers[-1].activation_value
    def grads(self,x,y,error_func):
        self.inference(x)
        delta=error_func(self.layers[-1].value,y)
        for i in reversed(range(1, len(self.layers))):
            delta=self.layers[i].back(self.layers[i-1].activation_value,delta)
    def update(self,learning_rate):
        for i in range(1,len(self.layers)):
            for j in range(0,self.layers[i-1].layer_size):
                for m in range(0,self.layers[i].layer_size):
                    self.layers[i].weights[j][m]-=self.layers[i].weight_grads[j][m]*learning_rate
                    self.layers[i].weight_grads[j][m]=0.0
            for m in range(0,self.layers[i].layer_size):
                self.layers[i].bias[m]-=self.layers[i].bias_grads[m]*learning_rate
                self.layers[i].bias_grads[m]=0.0
    def stochastic_gradient_descent(self, x, y, x_val, y_val, error_func, learning_rate, epochs, early_stopping_tolerance=5, verbose=True):
        loss_min=2**16
        c=0
        stop_early=False
        for e in range(epochs):
            for i in range(len(x)):
                self.grads(x[i],y[i],error_func)
                self.update(learning_rate)
                loss=0.0
                for j in range(len(x_val)):
                    y_pred=self.inference(x_val[j])
                    loss+=mse(y_pred,y_val[j],True)
                loss/=len(x_val)
                loss_min_orig=loss_min
                loss_min=min(loss,loss_min)
                if loss_min==loss_min_orig:
                    c+=1
                elif loss_min<loss_min_orig:
                    c=0
                if c==early_stopping_tolerance and early_stopping_tolerance!=0:
                    stop_early=True
                    break
                if verbose:
                    print("loss:",loss)
            if stop_early:
                break
