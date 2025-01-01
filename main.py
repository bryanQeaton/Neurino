import nn
x=[
    [-1,-1],
    [-1,1],
    [1,-1],
    [1,1]
]
y=[
    [0,1],
    [1,0],
    [1,0],
    [0,1]
]


layers=[
    nn.Model.InputLayer(2),
    nn.Model.Layer(3,nn.relu),
    nn.Model.Layer(3,nn.relu),
    nn.Model.Layer(2,nn.sigmoid)
]
model=nn.Model(layers)
model.stochastic_batch_gradient_descent(
    x, y, x, y,4,
    nn.mse, .01, 1000)



