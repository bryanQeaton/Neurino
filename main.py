import nn
x=[
    [0,0],
    [0,1],
    [1,0],
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
    nn.Model.Layer(3,nn.relu),
    nn.Model.Layer(1,nn.sigmoid)
]
model=nn.Model(layers)


