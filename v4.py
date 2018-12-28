import torch as t
from data import *
import pickle
from ml_helper.ml_helper import torch_helpers as mlt
import os
import time

data_folder = '/'.join(os.getcwd().split('/')[:-1] + ['data'])

tf = t.float
ti = t.int
device = t.device("cuda")

embedding_dim = 13
seq_len = 700

# create classes
max_class_count = 8
min_length = 50
peek = 3

v = lambda x: x.data.item()


class v33tgn01():
    def __init__(self, max_class_count, window_size, peek, lstm_layers=1, embedding_dim=4, time_scale=0.01):
        self.max_class_count = max_class_count
        self.one_hot = t.eye(max_class_count, device=device)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.peek = peek
        self.time_scale = time_scale
        self.lstm_layers = lstm_layers

        return

    def create_model(self):

        # weights
        self.w_node_emb = t.randn((max_class_count, embedding_dim), dtype=tf, device=device, requires_grad=True)

        # batch size 1 for ease of usability
        self.lstm_model, self.lstm_h_init = mlt.create_lstm(input_size=self.embedding_dim + 1,  # + x for each param
                                                            output_size=self.embedding_dim, batch_size=1,
                                                            num_of_layers=self.lstm_layers, device=device)

        self.e_lstm_out_to_h0, e1 = mlt.create_linear_layers(
            layer_sizes=[self.lstm_layers * embedding_dim * 2, 2000, 800, embedding_dim * 2],
            device=device)
        self.e_h0_to_p, e2 = mlt.create_linear_layers(layer_sizes=[embedding_dim * 2, 1], device=device)

        # learning_weights = [self.w_node_emb] + e1 + e2 + e3 + e4 + e5 + e6 + list(self.lstm_model.parameters())
        learning_weights = [self.w_node_emb] + e1 + e2 + list(self.lstm_model.parameters())

        self.learning_params = [
            {'params': learning_weights},
            # {'params': self.lstm_model.parameters()}
        ]
        return

    def train_model(self, data, batch_size=10, iter=100, lr=1e-3, save_path=None, bprop=True):
        _f = lambda x: t.tensor(x, device=device)
        data = [_f(x) for x in data]
        try:  # sorry, shouldnt be doing this way but its convenient
            self.w_node_emb
        except:
            self.create_model()

        self.optim = t.optim.SGD(self.learning_params, lr=lr)

        i = 0
        while True:
            batch_data = self.get_batch_data_by_rnd_label(data, batch_size)
            l, batch_loss = self.get_batch_loss(batch_data=batch_data, neg_sample_cnt=3)

            print('Batch loss:', l)
            if bprop:
                self._batch._batch_optimize(self, batch_loss)

            i += 1

            if save_path:
                if i % 10 == 0 and i != 0:
                    with open(save_path, 'wb') as file_object:  # save
                        pickle.dump(obj=self, file=file_object)
                        print('saved')
                    i = 0
        return

    def get_batch_data_by_rnd_label(self, data, batch_size):

        t.manual_seed(time.time())
        rnd_label = 4  # t.randint(self.max_class_count, (1, 1)).squeeze().data.item()
        pos_batch = []
        while (len(pos_batch) == 0):
            pos_batch = self._batch._create_training_batch_pos_for_label(label=rnd_label, data=data,
                                                                         batch_size=batch_size)
        print('Current training label:', rnd_label)
        batch_output = [self._fprop(x) for x in pos_batch]

        return batch_output

    def get_batch_loss(self, batch_data, neg_sample_cnt=5):
        classes = t.range(0, self.max_class_count, dtype=t.int64, device=device)
        batch_losses = [self._get_loss(d, neg_sample_cnt=neg_sample_cnt) for d in batch_data]
        return np.mean([x.data.item() for x in batch_losses]), batch_losses

    def _use_est(self, e, i, act=None):
        if act == None:
            act = lambda x: x
        o = act(e[0](i))
        for ee in e[1:]:
            o = act(ee(o))
        return o

    def _fprop(self, data):
        labels, timings, par_pos = data

        # check for orphans
        parent_pos = par_pos[-1]
        out = self._fprop_a(current_data=data, parent_pos=parent_pos)
        return out

    def _fprop_a(self, current_data, parent_pos):
        labels, timings, par_pos = current_data
        current_time = timings[-1]
        current_label = labels[-1]

        # check for orphans
        parent_label = labels[parent_pos]

        norm_timings = t.tanh(timings.sub(current_time).mul(self.time_scale))

        label_emb = t.index_select(self.w_node_emb, dim=0, index=labels)
        emb_time = t.cat([label_emb, norm_timings.unsqueeze(1)], dim=1).unsqueeze(1)

        c_hs = self.lstm_model(emb_time, self.lstm_h_init)[1]
        p_hs = self.lstm_model(emb_time[:parent_pos + 1], self.lstm_h_init)[1]

        current_emb = t.index_select(label_emb, 0, current_label).squeeze()
        parent_emb = t.index_select(label_emb, 0, parent_label).squeeze()

        _s = lambda x: x.reshape(self.lstm_layers * embedding_dim).squeeze()
        lstm_out = t.cat([_s(p_hs[1]), _s(c_hs[1])], dim=0)
        # lstm_out = t.cat([current_emb, parent_emb], dim=0)
        link = self._e_prop(self.e_lstm_out_to_h0, lstm_out, a=t.tanh)
        link = self._e_prop(self.e_h0_to_p, link, a=t.sigmoid)
        return {'link probability': link, 'parent': {'pos': parent_pos, 'label': labels[parent_pos]},
                'current': {'pos': len(labels), 'label': current_label}, 'data': current_data}

    def _get_loss(self, d, neg_sample_cnt=3):
        p_label = d['parent']['label']
        p_pos = d['parent']['pos']
        current_data = d['data']

        link_p = d['link probability']
        link_p_loss = link_p.sub(1).pow(2)
        bce_gtruth = t.ones(1, device=device)

        for n in range(neg_sample_cnt):
            # select non parents
            neg_parent_pos = t.randint(len(current_data[0]) - 1, (1, 1)).squeeze().data.item()
            neg_class = current_data[0][neg_parent_pos]

            while neg_class == p_label:
                neg_parent_pos = t.randint(len(current_data[0]) - 1, (1, 1)).squeeze().data.item()
                neg_class = current_data[0][neg_parent_pos]

            neg_out = self._fprop_a(current_data=current_data, parent_pos=neg_parent_pos)
            link_p = t.cat([link_p, neg_out['link probability']], dim=0)
            bce_gtruth = t.cat([bce_gtruth, t.zeros(1, device=device)])
        link_p = t.nn.Softmax()(link_p)
        bce_loss = t.nn.BCELoss()(link_p, bce_gtruth)
        return bce_loss  # link_p.sub(1).pow(2)

    @staticmethod
    def _e_prop(e, input, a=None):
        out = input
        for l in e:
            if a == None:
                out = l(out)
            else:
                out = a(l(out))
        return out

    class _batch():
        # create training data
        @staticmethod
        def _create_training_batch_pos_for_label(data, label, batch_size):
            seq_class_data = data[0]
            label_pos = (seq_class_data == label).nonzero()

            # filters out those that are too short
            pos_filter = label_pos - min_length + 1 > 0
            label_pos = t.masked_select(label_pos, pos_filter)
            pos_filter = label_pos + peek + 1 < len(seq_class_data)
            label_pos = t.masked_select(label_pos, pos_filter)

            # shuffle
            label_pos = label_pos[t.randperm(label_pos.shape[0])]
            if label_pos.shape[0] > batch_size:
                label_pos = label_pos[:batch_size]
            batch_data = [[data[0][:x], data[1][:x], data[2][:x]] for x in label_pos]
            return batch_data

        @staticmethod
        def _batch_fprop(self, batch):
            # for training only
            batch_output = []
            for d in batch:
                batch_output.append(self._fprop(d))
            return batch_output

        @staticmethod
        def _batch_optimize(self, batch_loss):
            [x.backward(retain_graph=True) for x in batch_loss]
            self.optim.step()
            self.optim.zero_grad()
            return


data_map_path = data_folder + '/dataV4'
save_path = data_folder + '/modelV4'

diagnostics_mode = False

save_mode = True
load_from_old = True

if diagnostics_mode:
    save_mode, load_from_old = False, True

data = []
dg = data_generator()
if load_from_old:
    if save_mode:
        data = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                                save_path=data_map_path, load_path=data_map_path)
    else:
        data = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                                load_path=data_map_path)
elif save_mode:
    data = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                            save_path=data_map_path)
else:
    data = dg.generate_data(seq_len=seq_len, class_count=max_class_count)

# seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count)

model = v33tgn01(time_scale=0.001, max_class_count=max_class_count,
                 window_size=min_length, peek=peek,
                 embedding_dim=embedding_dim, lstm_layers=3)
if load_from_old:
    if os.path.exists(save_path):
        with open(save_path, 'rb') as file_object:  # load
            model = pickle.load(file_object)
if not diagnostics_mode:
    model.train_model(data, save_path=save_path, batch_size=300, lr=1e-6)
