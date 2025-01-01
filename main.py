import nn
x=[
    [-1,-1],
    [-1,1],
    [1,-1],
    [1,1]
]
y=[
    [0],
    [1],
    [1],
    [0]
]

layers=[
    nn.Model.InputLayer(2),
    nn.Model.Layer(20,nn.relu),
    nn.Model.Layer(1,nn.linear)
]
model=nn.Model(layers)
model.stochastic_batch_gradient_descent(
    x, y, x, y,4,
    nn.mse, .01,1000,10,True)
for item in x:
    print(model.inference(item))



