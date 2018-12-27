import time
import torch as t
from data import *
import pickle
from ml_helper.ml_helper import torch_helpers as mlt
import os
import copy

tf = t.float
ti = t.int
device = t.device("cpu")

embedding_dim = 50
seq_len = 700

# create classes
max_class_count = 8
window_size = 15
peek = 3

v = lambda x: x.data.item()


class v33tgn01():
    def __init__(self, max_class_count, window_size, peek, lstm_layers=1, embedding_dim=4, time_scale=0.01):
        self.max_class_count = max_class_count
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.peek = peek
        self.time_scale = time_scale
        self.lstm_layers = lstm_layers

        return

    def create_model(self):
        # batch size 1 for ease of usability
        self.lstm_model, self.lstm_h_init = mlt.create_lstm(input_dim=self.embedding_dim,
                                                            output_size=self.embedding_dim, batch_size=1,
                                                            num_of_layers=self.lstm_layers, device=device)
        lstm_hidden_param_size = self.lstm_h_init[0].reshape(-1).shape[0]

        # weights
        self.w_node_emb = t.randn((max_class_count, embedding_dim), dtype=tf, device=device, requires_grad=True)
        self.w_time_emb = t.randn((1, embedding_dim), dtype=tf, device=device, requires_grad=True)

        # or you can just use sum()
        self.w_node_time_aff = t.randn((embedding_dim * 2, embedding_dim), dtype=tf, device=device, requires_grad=True)

        self.w_output_emb_1 = t.randn((window_size * embedding_dim, 20), dtype=tf,
                                      device=device, requires_grad=True)
        # output_emb not converging, adding extra layers
        self.w_output_emb_2 = t.randn((20, (window_size) * embedding_dim), dtype=tf,
                                      device=device, requires_grad=True)

        self.w_output_time = t.randn((window_size * embedding_dim, (window_size)), dtype=tf, device=device,
                                     requires_grad=True)

        # all things future need lstm output
        self.w_future_emb = t.randn((window_size * embedding_dim + 2 * lstm_hidden_param_size, (peek) * embedding_dim),
                                    dtype=tf, device=device,
                                    requires_grad=True)

        self.w_future_time = t.randn((window_size * embedding_dim + 2 * lstm_hidden_param_size, (peek)), dtype=tf,
                                     device=device, requires_grad=True)

        # biases
        self.b_node_time_aff = t.randn((embedding_dim), dtype=tf, device=device, requires_grad=True)

        self.b_output_emb_1 = t.randn((20), dtype=tf, device=device, requires_grad=True)
        self.b_output_emb_2 = t.randn(((window_size) * embedding_dim), dtype=tf, device=device, requires_grad=True)

        self.b_output_time = t.randn((window_size), dtype=tf, device=device, requires_grad=True)

        self.b_future_emb = t.randn(((peek) * embedding_dim), dtype=tf, device=device,
                                    requires_grad=True)

        self.b_future_time = t.randn((peek), dtype=tf, device=device, requires_grad=True)

        learning_weights = [
            self.w_node_emb,
            self.w_time_emb,
            self.w_output_time,
            self.w_output_emb_1,
            self.w_output_emb_2,
            self.w_node_time_aff,
            self.w_future_emb,
            self.w_future_time
        ]

        learning_biases = [
            self.b_output_time,
            self.b_output_emb_1,
            self.b_output_emb_2,
            self.b_node_time_aff,
            self.b_future_emb,
            self.b_future_time
        ]

        self.learning_params = [
            {'params': learning_weights},
            {'params': learning_biases},
            {'params': self.lstm_model.parameters()}
        ]
        return

    def train_model(self, seq_class_data, seq_time_data, batch_size=10, iter=100, lr=1e-3,
                    save_path=None, bprop=True):
        try:  # sorry, shouldnt be doing this way but its convenient
            self.w_node_emb
        except:
            self.create_model()

        self.optim = t.optim.Adam(self.learning_params, lr=lr)

        i = 0
        while True:
            l, batch_loss = self.get_batch_loss(seq_class_data=seq_class_data, seq_time_data=seq_time_data,
                                                batch_size=batch_size)

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

    def get_batch_loss(self, seq_class_data, seq_time_data, batch_size):
        seq_class_data = t.tensor(seq_class_data, device=device)
        seq_time_data = t.tensor(seq_time_data, device=device)

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
        batch_negative_data = self._batch._batch_add_noise(self, batch_data, noise_ratio=1)

        batch_embedding_output = self._batch._batch_fprop(self, batch_data)
        batch_loss = self._batch._batch_loss(self, batch_embedding_output, batch_data, batch_negative_data)
        l = t.tensor(([t.tensor(x).mean() for x in batch_loss])).mean().data.item()
        return l, batch_loss

    def run_diagnostics(self, original_seq_class_data, original_seq_time_data, batch_size):
        original_seq_class_data = t.tensor(original_seq_class_data, device=device)
        original_seq_time_data = t.tensor(original_seq_time_data, device=device)

        noise_levels = np.linspace(1, 0, 6)
        for noise in noise_levels:
            print('Testing against ', round(noise * 100), '% noise...')
            t.manual_seed(time.time())
            rnd_label = t.randint(self.max_class_count, (1, 1)).squeeze().data.item()
            print('Current training label:', rnd_label)

            _h = lambda x: self._batch._create_training_batch_for_label(self, label=x,
                                                                        seq_class_data=original_seq_class_data,
                                                                        seq_time_data=original_seq_time_data,
                                                                        batch_size=batch_size,
                                                                        window_size=window_size,
                                                                        peek=peek)

            batch_data = _h(rnd_label)
            batch_negative_data = self._batch._batch_add_noise(self, batch_data, noise_ratio=noise)

            batch_embedding_output = self._batch._batch_fprop(self, batch_data)
            batch_loss = self._batch._batch_loss(self, batch_embedding_output, batch_data, batch_negative_data,
                                                 diagnostics_mode=True)
            diag_data = {}
            all_keys = [k for k in batch_loss[0][0].keys()]

            for k in all_keys:
                diag_data[k] = []

            for b in batch_loss:
                for k in all_keys:
                    key_values = [x[k] for x in b]
                    diag_data[k] += key_values

            report = ''
            for k in all_keys:
                report += k + ' mean:' + str(np.mean(diag_data[k]))[:4] + ' std:' + str(np.std(diag_data[k]))[:4] + ' '
            print(report)

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
                batch_output.append(self._fprop(d))
            return batch_output

        @staticmethod
        def _batch_loss(self, batch_output, batch, neg_batch, diagnostics_mode=False):
            def _xent_loss(a, b, truth_array):
                l = b.add(a).mean(dim=1).sigmoid()
                return t.nn.BCELoss()(input=l,
                                      target=truth_array)  # t.nn.BCEWithLogitsLoss()(input=l, target=truth_array)

            batch_loss = []

            for i in range(len(batch)):
                frame_cnt = batch[i]['frame_count']

                p_time_list = [x[0] for x in batch_output[i]]
                p_emb_list = [x[1] for x in batch_output[i]]

                y_time_list = batch[i]['x']['time']
                y_class_list = batch[i]['x']['class']

                neg_time_list = neg_batch[i]['x']['time']
                neg_class_list = neg_batch[i]['x']['class']

                frame_loss = []
                diagnostics_loss = []

                for fc in range(frame_cnt):
                    time_loss = p_time_list[fc].add(-y_time_list[fc]).pow(2).mean()
                    # time_neg_loss = p_time_list[fc].add(neg_time_list[fc]).pow(2).mean()
                    # time_loss = time_pos_loss.add(time_neg_loss).mean()

                    py_emb = p_emb_list[fc].reshape(window_size, -1)
                    y_class, neg_class = y_class_list[fc], neg_class_list[fc]
                    pos_truth = t.ones(window_size, device=device)
                    neg_truth = t.where(y_class == neg_class, pos_truth, t.zeros(window_size, device=device))

                    y_emb = t.index_select(self.w_node_emb, dim=0, index=y_class)
                    true_xent = _xent_loss(py_emb, y_emb, truth_array=pos_truth)
                    neg_emb = t.index_select(self.w_node_emb, dim=0, index=neg_class_list[fc])
                    neg_xent = _xent_loss(py_emb, neg_emb, truth_array=neg_truth)

                    emb_loss = true_xent.sum().add(neg_xent.sum())

                    frame_loss.append(time_loss.add(emb_loss))
                    if diagnostics_mode:
                        diagnostics_loss.append(
                            {'Time loss': v(time_loss), 'True xent loss': v(true_xent), 'Neg xent loss': v(neg_xent),
                             'Emb loss': v(emb_loss)})
                        # frame_loss.append(emb_loss)

                # future loss
                p_future_time = p_time_list[-1]
                y_future_time = batch[i]['y']['time'][-1]
                neg_future_time = neg_batch[i]['y']['time'][-1]
                future_time_loss = p_future_time.add(-y_future_time).pow(2).mean()
                # future_time_neg_loss = p_future_time.add(neg_future_time).pow(2).mean()
                # future_time_loss = future_time_pos_loss.add(future_time_neg_loss)

                future_y, future_neg = batch[i]['y']['class'][-1], neg_batch[i]['y']['class'][-1]
                future_py_emb = p_emb_list[-1].reshape(peek, -1)
                future_y_emb = t.index_select(self.w_node_emb, dim=0, index=future_y)

                future_pos_truth = t.ones(peek, device=device)
                future_neg_truth = t.where(future_y == future_neg, future_pos_truth,
                                           t.zeros(peek, device=device))

                future_true_xent = _xent_loss(future_py_emb, future_y_emb, truth_array=future_pos_truth)
                future_neg_emb = t.index_select(self.w_node_emb, dim=0, index=future_neg)
                future_neg_xent = _xent_loss(future_py_emb, future_neg_emb, truth_array=future_neg_truth)

                future_emb_loss = future_true_xent.sum().add(future_neg_xent)

                if diagnostics_mode:
                    diagnostics_loss.append(
                        {'Time loss': v(future_time_loss), 'True xent loss': v(future_true_xent),
                         'Neg xent loss': v(future_neg_xent), 'Emb loss': v(future_emb_loss)})

                future_loss = future_emb_loss.add(future_time_loss)
                # future_loss = future_emb_loss

                frame_loss.append(future_loss)
                if not diagnostics_mode:
                    batch_loss.append(frame_loss)
                else:
                    batch_loss.append(diagnostics_loss)
            return batch_loss

        @staticmethod
        def _batch_optimize(self, batch_loss):
            [item.backward(retain_graph=True) for sublist in batch_loss for item in sublist]
            self.optim.step()
            self.optim.zero_grad()
            return

        @staticmethod
        def _batch_add_noise(self, batcho, noise_ratio=1):
            batch = self._batch._clone_batch(batcho)

            def _i(int_t, float_t, class_cnt, noise):
                p_rnd = t.rand((len(int_t)), device=device)

                int_rnd = t.randint(class_cnt, (len(int_t), 1), device=device).squeeze()
                noisy_int = t.where(p_rnd > 1 - noise, int_rnd, int_t)

                float_rnd = t.rand((len(float_t)), device=device).mul(2).add(-1)
                noisy_float = t.where(p_rnd > 1 - noise, float_rnd, float_t)
                return noisy_int, noisy_float

            def _h(d, l, i):
                cl, ti = d[l]['class'][i], d[l]['time'][i]
                return _i(int_t=cl, float_t=ti, class_cnt=self.max_class_count, noise=noise_ratio)

            for i in range(len(batch)):
                d = batch[i]

                for f in range(d['frame_count']):
                    try:
                        d['x']['class'][f], d['x']['time'][f] = _h(d, 'x', f)
                        d['y']['class'][f], d['y']['time'][f] = _h(d, 'y', f)
                    except:
                        raise Exception('Error')
            return batch

        @staticmethod
        def _clone_batch(batch):
            b_ = []

            def _h(ct):
                ct_ = {}
                ct_['class'] = [x.clone().detach() for x in ct['class']]
                ct_['time'] = [x.clone().detach() for x in ct['time']]
                return ct_

            for d in batch:
                b_.append({'x': _h(d['x']), 'y': _h(d['y']), 'pos': d['pos'], 'frame_count': d['frame_count']})
            return b_

    def create_data_point(self, pos, seq_class_data, seq_time_data):
        def _h(dset, pos, denom):
            r = t.remainder(pos, denom)
            o = dset[r + 1:pos + 1]
            x_set = o.split(denom)
            y_set = [x[:self.peek] for x in x_set[1:]]
            y_set += [dset[pos + 1:pos + peek + 1]]
            return list(x_set), y_set, len(x_set)

        # splits all data before current label position into train sets for unlimited len lstm input
        pos_xy_data_set = _h(seq_class_data, pos, self.window_size)

        l_seq_time_data = seq_time_data[:pos + self.peek + 1].add(-seq_time_data[pos])
        l_seq_time_data = t.tanh(t.mul(l_seq_time_data, self.time_scale))
        pos_xy_time_set = _h(l_seq_time_data, pos, self.window_size)

        x = {'class': pos_xy_data_set[0], 'time': pos_xy_time_set[0]}
        y = {'class': pos_xy_data_set[1], 'time': pos_xy_time_set[1]}

        return {'x': x, 'y': y, 'pos': pos, 'frame_count': pos_xy_time_set[2]}

    def _fprop(self, d):
        output = []

        # uses only x
        frame_count = d['frame_count']
        h_lstm = self.lstm_h_init

        def _f(xd, xt):
            seq_emb = t.index_select(self.w_node_emb, dim=0, index=xd)
            data_time_emb = xt.unsqueeze(1).mm(self.w_time_emb)
            seq_time_cat_emb = t.cat((seq_emb, data_time_emb), dim=1)

            combined_aff = t.tanh(seq_time_cat_emb.mm(self.w_node_time_aff).add(self.b_node_time_aff))
            return combined_aff

        for i in range(frame_count):
            xd = d['x']['class'][i]
            xt = d['x']['time'][i]
            d_emb = _f(xd=xd, xt=xt).unsqueeze(1)  # unsqueeze for batch size(1)
            d_lstm_emb, h_lstm = self.lstm_model(d_emb, h_lstm)

            # need to train for every frag
            d_lstm_emb = d_lstm_emb.reshape((1, self.window_size * self.embedding_dim))
            out_time = t.tanh(d_lstm_emb.mm(self.w_output_time).add(self.b_output_time))

            out_emb = t.tanh(d_lstm_emb.mm(self.w_output_emb_1).add(self.b_output_emb_1))
            out_emb = t.tanh(out_emb.mm(self.w_output_emb_2).add(self.b_output_emb_2))

            output.append([out_time, out_emb, h_lstm])

            if i == frame_count - 1:
                # last one add future data
                hidden_cat = t.cat((d_lstm_emb, h_lstm[0].reshape(1, -1), h_lstm[1].reshape(1, -1)), dim=1)
                future_time = t.tanh(hidden_cat.mm(self.w_future_time).add(self.b_future_time))
                future_emb = t.tanh(hidden_cat.mm(self.w_future_emb).add(self.b_future_emb))
                output.append([future_time, future_emb, h_lstm])
        return output


data_map_path = 'data_v2'
save_path = 'model_v2'

diagnostics_mode = False

save_mode = True
load_from_old = False

if diagnostics_mode:
    save_mode, load_from_old = False, True

seq_data_, seq_time_data_ = [], []
dg = data_generator()
if load_from_old:
    if save_mode:
        seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                                                     save_path=data_map_path, load_path=data_map_path)
    else:
        seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                                                     load_path=data_map_path)
elif save_mode:
    seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count,
                                                 save_path=data_map_path)
else:
    seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count)

# seq_data_, seq_time_data_ = dg.generate_data(seq_len=seq_len, class_count=max_class_count)

model = v33tgn01(time_scale=0.001, max_class_count=max_class_count,
                 window_size=window_size, peek=peek,
                 embedding_dim=embedding_dim, lstm_layers=3)
if load_from_old:
    if os.path.exists(save_path):
        with open(save_path, 'rb') as file_object:  # load
            model = pickle.load(file_object)
if not diagnostics_mode:
    model.train_model(seq_class_data=seq_data_, seq_time_data=seq_time_data_, save_path=save_path, batch_size=10,
                      lr=1e-6)

else:
    model.run_diagnostics(original_seq_class_data=seq_data_, original_seq_time_data=seq_time_data_, batch_size=10)
