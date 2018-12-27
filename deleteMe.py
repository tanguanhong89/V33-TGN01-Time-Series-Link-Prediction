import numpy as np
import torch as t

x = np.random.rand(100, 2)
y = np.zeros((100, 1))

for i in range(100):
    y[i] = 1 if np.sum(x[i]) > 1 else 0

x = t.tensor(x, dtype=t.float32)
y = t.tensor(y, dtype=t.int64).squeeze()

cc = t.eye(2)
y = cc.index_select(0, y)

l1 = t.nn.Linear(2, 100)
l2 = t.nn.Linear(100, 2)

opt = t.optim.SGD(list(l1.parameters()) + list(l2.parameters()), lr=0.01)


def lf(y_, y):
    pos = (y_.sub(y)).pow(2).mean()
    neg = (y_.sub(y.sub(1).abs())).pow(2).mean()
    return pos.div(neg)


def lf2(y_, y):
    pos = (y_.sub(y)).pow(2).mean()
    return pos


for i in range(100):
    y_ = t.sigmoid(l2(t.tanh(l1(x))))

    loss = t.nn.BCELoss()(input=y_, target=y)

    # loss = lf(y_, y)
    loss.backward()

    opt.step()
    opt.zero_grad()
    print('loss:', loss, ' wrong:', y_.round().sub(y).pow(2).sum())

ad=5