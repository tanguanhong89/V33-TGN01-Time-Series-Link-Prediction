import time
import torch as t
from data import *
import pickle
from ml_helper.ml_helper import torch_helpers as mlt
import os

tf = t.float
ti = t.int
device = t.device("cpu")

input_feature_cnt = 2
embedding_dim = 4
seq_len = 1000

# create classes
max_class_count = 10
window_size = int(seq_len / 20)
peek = 5


class v33tgn01():
    def __init__(self, input_dim, max_class_count, window_size, peek, lstm_layers=1, embedding_dim=4, time_scale=0.01):
        self.input_dim = input_dim
        self.max_class_count = max_class_count
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.peek = peek
        self.time_scale = time_scale
        self.lstm_layers = lstm_layers

        return

    def create_model(self):
        # weights
        self.w_node_embeddings = t.randn((max_class_count, embedding_dim), dtype=tf, device=device, requires_grad=True)
        self.w_time_embeddings = t.randn((1, embedding_dim), dtype=tf, device=device, requires_grad=True)

        # or you can just use sum()
        self.w_aff_embeddings = t.randn((1, embedding_dim), dtype=tf, device=device, requires_grad=True)

        self.w_output_embeddings = t.randn((window_size * embedding_dim, (window_size + peek) * embedding_dim),
                                           dtype=tf,
                                           device=device,
                                           requires_grad=True)
        self.w_output_time = t.randn((window_size * embedding_dim, (window_size + peek)), dtype=tf, device=device,
                                     requires_grad=True)

        # batch size 1 for ease of usability
        self.lstm_model, self.lstm_h_init = mlt.create_lstm(input_size=self.embedding_dim,
                                                            output_size=self.embedding_dim, batch_size=1,
                                                            num_of_layers=self.lstm_layers, device=device)
        learning_params = [
            self.w_node_embeddings,
            self.w_time_embeddings,
            self.w_output_time,
            self.w_output_embeddings,
            self.w_aff_embeddings
        ]

        self.learning_params = [
            {'params': learning_params},
            {'params': self.lstm_model.parameters()}
        ]
        return

    def train_model(self, seq_class_data, seq_time_data, batch_size=10, iter=100, lr=1e-3, save_path=None):
        seq_class_data = t.tensor(seq_class_data, device=device)
        seq_time_data = t.tensor(seq_time_data, device=device)

        try:  # sorry, shouldnt be doing this way but its convenient
            self.w_node_embeddings
        except:
            self.create_model()

        self.optim = t.optim.Adam(self.learning_params, lr=lr)

        i = 0
        while True:
            t.manual_seed(time.time())
            rnd_label = t.randint(self.max_class_count, (1, 1)).squeeze().data.item()
            print('Current training label:', rnd_label)

            _h = lambda x: self._batch._create_training_batch_for_label(self, label=x,
                                                                        seq_class_data=seq_class_data,
                                                                        seq_time_data=seq_time_data,
                                                                        batch_size=batch_size,
                                                                        window_size=window_size,
                                                                        peek=peek)

            batch_data = _h(rnd_label)

            rnd_negative_label = t.randint(self.max_class_count, (1, 1)).squeeze().data.item()
            while rnd_negative_label == rnd_label:
                rnd_negative_label = t.randint(self.max_class_count, (1, 1)).squeeze().data.item()

            batch_negative_data = _h(rnd_negative_label)

            batch_embedding_output = self._batch._batch_fprop(self, batch_data)
            batch_loss = self._batch._batch_loss(self, batch_embedding_output, batch_data, batch_negative_data)
            print('Batch loss:', t.tensor(batch_loss).mean().data.item())
            self._batch._batch_optimize(self, batch_loss)

            i += 1

            if save_path:
                if i % 10 == 0:
                    with open(save_path, 'wb') as file_object:  # save
                        pickle.dump(obj=self, file=file_object)
                    i = 0

        return

    class _batch():
        # create training data
        @staticmethod
        def _create_training_batch_for_label(self, seq_class_data, seq_time_data, label, batch_size, window_size, peek):
            label_pos = (seq_class_data == label).nonzero()
            # filters out those that are too short
            pos_filter = label_pos - window_size + 1 > 0
            label_pos = t.masked_select(label_pos, pos_filter)
            pos_filter = label_pos + peek + 1 < len(seq_class_data)
            label_pos = t.masked_select(label_pos, pos_filter)

            # shuffle
            label_pos = label_pos[t.randperm(label_pos.shape[0])]

            if label_pos.shape[0] > batch_size:
                label_pos = label_pos[:batch_size]

            batch = []

            for pos in label_pos:
                o = self.create_data_point(pos=pos, seq_class_data=seq_class_data, seq_time_data=seq_time_data)
                batch.append(o)
            return batch

        @staticmethod
        def _batch_fprop(self, batch):
            # for training only
            batch_output = []
            for d in batch:
                out_time, out_node_embeddings = self._fprop(d)
                batch_output.append([out_time, out_node_embeddings])
            return batch_output

        @staticmethod
        def _batch_loss(self, batch_output, batch, neg_batch):
            batch_loss = []

            def _h(d):
                time = t.cat((d['x']['time'][-1], d['y']['time'][-1]))

                y_class = t.cat((d['x']['class'][-1], d['y']['class'][-1]))
                node_embeddings = t.index_select(self.w_node_embeddings, dim=0,
                                                 index=y_class)
                return time, node_embeddings.reshape(-1)

            for i in range(len(batch)):
                out_time = batch_output[i][0]
                out_node_emb = batch_output[i][1]

                pos = batch[i]
                pos_time, pos_node_embeddings = _h(pos)

                neg = neg_batch[i]
                neg_time, neg_node_embeddings = _h(neg)

                feed_dict = {
                    'node': {
                        'time': out_time,
                        'emb': out_node_emb
                    },
                    'pos': {
                        'time': pos_time,
                        'emb': pos_node_embeddings
                    },
                    'neg': {
                        'time': neg_time,
                        'emb': neg_node_embeddings
                    }
                }

                loss = self._embed_loss(feed_dict)
                batch_loss.append(loss)
            return batch_loss

        @staticmethod
        def _batch_optimize(self, batch_loss):
            for i in batch_loss:
                i.backward(retain_graph=True)
            self.optim.step()
            self.optim.zero_grad()
            return

    def create_data_point(self, pos, seq_class_data, seq_time_data):
        def _h(dset, pos, denom):
            r = t.remainder(pos, denom)
            o = dset[r + 1:pos + 1]
            x_set = o.split(denom)
            y_set = [x[:self.peek] for x in x_set[1:]]
            y_set += [dset[pos + 1:pos + peek + 1]]
            return x_set, y_set, len(x_set)

        # splits all data before current label position into train sets for unlimited len lstm input
        pos_xy_data_set = _h(seq_class_data, pos, self.window_size)

        l_seq_time_data = seq_time_data[:pos + self.peek + 1].add(-seq_time_data[pos])
        l_seq_time_data = t.tanh(t.mul(l_seq_time_data, self.time_scale))
        pos_xy_time_set = _h(l_seq_time_data, pos, self.window_size)

        x = {'class': pos_xy_data_set[0], 'time': pos_xy_time_set[0]}
        y = {'class': pos_xy_data_set[1], 'time': pos_xy_time_set[1]}

        return {'x': x, 'y': y, 'pos': pos, 'frame_count': pos_xy_time_set[2]}

    def _fprop(self, d):
        # uses only x
        frame_count = d['frame_count']
        h_lstm = self.lstm_h_init

        def _f(xd, xt):
            data_node_embeddings = t.tanh(t.index_select(self.w_node_embeddings, dim=0, index=xd))
            data_time_embeddings = xt.unsqueeze(1).mm(self.w_time_embeddings)

            combined_aff_embeddings = data_node_embeddings.add(data_time_embeddings).pow(2)
            combined_aff_embeddings = self.w_aff_embeddings.mul(combined_aff_embeddings)
            return combined_aff_embeddings

        for i in range(frame_count):
            xd = d['x']['class'][i]
            xt = d['x']['time'][i]
            d_emb = _f(xd=xd, xt=xt).unsqueeze(1)  # unsqueeze for batch size(1)
            d_lstm_emb, h_lstm = self.lstm_model(d_emb, h_lstm)

        d_lstm_emb = d_lstm_emb.reshape((1, self.window_size * self.embedding_dim))
        out_time = t.tanh(d_lstm_emb.mm(self.w_output_time))
        out_node_embeddings = d_lstm_emb.mm(self.w_output_embeddings).squeeze()
        return out_time, out_node_embeddings

    def _embed_loss(self, feed_dict):
        node_time = feed_dict['node']['time']
        node_emb = feed_dict['node']['emb']

        pos_time = feed_dict['pos']['time']
        pos_emb = feed_dict['pos']['emb']

        neg_time = feed_dict['neg']['time']
        neg_emb = feed_dict['neg']['emb']

        loss_time_reconstruct = node_time.add(-pos_time).pow(2).mean()
        loss_emb_reconstruct = node_emb.add(-pos_emb).pow(2).mean()

        loss_time_neg = node_time.add(neg_time).pow(2).mean()
        loss_emb_neg = node_emb.add(neg_emb).pow(2).mean()

        loss_sum = loss_time_reconstruct.add(loss_emb_reconstruct).add(loss_time_neg).add(
            loss_emb_neg)  # this assumes time and embed have equal importance (1:1)
        return loss_sum


scenerio = [1, 2, 3]

relationship_path = 'data_v0'
save_path = 'model'

save_mode = True
load_from_old = True

seq_data_, seq_time_data_ = [], []
if load_from_old:
    seq_data_, seq_time_data_ = generate_data(max_class_count=max_class_count, seq_len=seq_len,
                                              load_path=relationship_path)
elif save_mode:
    seq_data_, seq_time_data_ = generate_data(max_class_count=max_class_count, seq_len=seq_len,
                                              save_path=relationship_path)

model = v33tgn01(input_dim=input_feature_cnt, max_class_count=max_class_count, window_size=window_size, peek=peek,
                 embedding_dim=embedding_dim)
if load_from_old:
    if os.path.exists(save_path):
        with open(save_path, 'rb') as file_object:  # load
            model = pickle.load(file_object)
            a = 5

model.train_model(seq_class_data=seq_data_, seq_time_data=seq_time_data_, save_path=save_path)
