import torch as t
import numpy as np
import time


x = np.random.rand(100, 2)
y = np.zeros(100)
for i in range(len(x)):
    y[i] = 1 if np.sum(x[i]) > 1 else 0

x = t.tensor(x, dtype=t.float32)
y = t.tensor(y, dtype=t.int64)

oh = t.eye(2)
y = oh.index_select(0, y)

L1 = t.nn.Linear(2, 1000)
L2 = t.nn.Linear(1000, 2)

opt = t.optim.Adam(list(L1.parameters()) + list(L2.parameters()), lr=0.01)


def closs(y, y_):
    return y.sub(y_).pow(2).mean()


for i in range(100):
    y_ = t.nn.Softmax()(L2(L1(x)).sigmoid())
    t1 = time.time()
    #bceloss = t.nn.BCELoss()(y_, y)
    t2 = time.time()
    #bceloss.backward(retain_graph=True)
    t3 = time.time()

    cus_loss = closs(y_, y)
    t4 = time.time()
    cus_loss.backward(retain_graph=True)
    t5 = time.time()

    # print('BCE:', str(t2 - t1), '    BCE bw:', str(t3 - t2), '   cL:', str(t4 - t3), '   cL bw:', str(t5 - t4))
    wrong = y_.round().sub(y).abs().sum().data.item()
    print(wrong)
    opt.step()
    opt.zero_grad()
