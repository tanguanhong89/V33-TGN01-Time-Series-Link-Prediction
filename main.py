import os
from data_model import *
from time_series_link_prediction_model import v33tgn01
import torch as t

# the folder dir is for linux, change / to \ for windows
data_folder = os.getcwd() + '/data'  # '/'.join(os.getcwd().split('/')[:-1] + ['data'])
device = 'cpu'


def c(d):
    return [x.oRecordData for x in d]


db_name = '*'
usr = "*"
pw = "*"
host = "*"

data_model = ParentOfData(usr=usr, pw=pw, host=host, db=db_name)
save_data = data_folder + '/data'
save_model = data_folder + '/model'
if not os.path.exists(save_data):
    data_model.retrieve_data()
    data_model.export_data(save_data)
else:
    data_model = data_model.import_data(save_data)

save = False
load_from_old = True
batch_size = 20
iter = 100
lr = 1e-2

etc_fields = []
data = data_model.construct_sequential(etc=etc_fields)

model = v33tgn01(time_scale=1e-6, max_label_count=len(data_model.map), embedding_dim=8, etc_dim=len(etc_fields),
                 lstm_layers=2, device=device)
if load_from_old:
    if os.path.exists(save_model):
        with open(save_model, 'rb') as file_object:  # load
            model = t.load(file_object, map_location=device)
            print('Model loaded')
model.device = device
if save:
    model.train_model(data, batch_size=batch_size, lr=lr, iter=iter, bprop=True, save_path=save_model)
else:
    model.train_model(data, batch_size=batch_size, lr=lr, iter=iter, bprop=True)
