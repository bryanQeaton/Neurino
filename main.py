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

l=.01
epochs=5

layers=[
    nn.Model.InputLayer(2),
    nn.Model.Layer(3,nn.relu),
    nn.Model.Layer(1,nn.sigmoid)
]
model=nn.Model(layers)

for e in range(epochs):
    for i in range(len(x)):
        model.grads(x[i],y[i],nn.mse)
        print(nn.mse(model.inference(x[i]),y[i],True))
    model.update(l)







