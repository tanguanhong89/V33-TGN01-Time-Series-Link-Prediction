import torch as t
from ml_helper.ml_helper import torch_helpers as mlt
import os
import time
import numpy as np

data_folder = '/'.join(os.getcwd().split('/')[:-1] + ['data'])

tf = t.float32
ti = t.int64

v = lambda x: x.data.item()


class v33tgn01():
    def __init__(self, max_label_count, lstm_layers=1, embedding_dim=4, etc_dim=0, time_scale=0.01, device='cpu'):
        self.max_label_count = max_label_count
        self.device = device
        self.all_labels = t.arange(0, self.max_label_count, device=self.device)
        self.one_hot = t.eye(max_label_count, device=self.device)
        self.embedding_dim = embedding_dim
        self.etc_dim = etc_dim
        self.time_scale = time_scale
        self.lstm_layers = lstm_layers

        return

    def create_model(self):
        label_cnt = self.max_label_count
        emb_dim = self.embedding_dim
        # weights
        self.w_node_emb = t.randn((label_cnt, emb_dim), dtype=tf, device=self.device, requires_grad=True)

        # batch size 1 for ease of usability
        lstm_input_dim = emb_dim + 1 + self.etc_dim
        self.lstm_model, self.lstm_h_init = mlt.create_lstm(input_size=lstm_input_dim,  # + x for each param
                                                            output_size=self.embedding_dim, batch_size=1,
                                                            num_of_layers=self.lstm_layers, device=self.device)
        # self.lstm_layers * emb_dim * 2 + emb_dim * 2
        self.e_lstm_out_to_h0, e1 = mlt.create_linear_layers(
            layer_sizes=[self.lstm_layers * emb_dim * 2 + emb_dim * 2, 2000, 1000, 500, emb_dim * 2],
            device=self.device)
        self.e_h0_to_p, e2 = mlt.create_linear_layers(layer_sizes=[emb_dim * 2, 2], device=self.device)

        # learning_weights = [self.w_node_emb] + e1 + e2 + e3 + e4 + e5 + e6 + list(self.lstm_model.parameters())
        learning_weights = [self.w_node_emb] + e1 + e2 + list(self.lstm_model.parameters())

        self.learning_params = [
            {'params': learning_weights},
            # {'params': self.lstm_model.parameters()}
        ]
        return

    def train_model(self, data, batch_size=10, lr=1e-3, save_path=None, bprop=True, iter=100):
        try:  # sorry, shouldnt be doing this way but its convenient
            self.w_node_emb
        except:
            self.create_model()

        _f = lambda x, y: t.tensor(x, dtype=y, device=self.device)
        seq_label = _f(data[0], ti)
        seq_time = _f(data[1], tf)
        seq_par = _f(data[2], ti)
        seq_etc = self._process_etc(data[3])

        last_loss = []

        data = [seq_label, seq_time, seq_par, seq_etc]

        self.optim = t.optim.SGD(self.learning_params, lr=lr)
        loss_tracker = []
        i = 0
        noise_steps = np.linspace(1, 0.01, 7)
        iter_multiplier = lambda x: np.exp(0.35 / (x + 0.2)) - 0.8
        for noise in noise_steps:
            print('Training noise @ ', str(noise))
            it_new = int(iter_multiplier(noise) * iter)
            for i in range(it_new):
                print('Iteration @ ', str(i), ' of ', str(it_new))
                try:
                    batch_data = self.get_batch_data_by_rnd_label(data, batch_size, mode='multi')
                    l, batch_loss = self.get_batch_loss(batch_data=batch_data)
                    l_neg, batch_loss_neg = self.get_neg_batch_loss(batch_data=[x['data'] for x in batch_data],
                                                                    noise=noise)
                    l = np.mean([l, l_neg])
                    batch_loss += batch_loss_neg
                    print('Batch loss(median):', l)
                    loss_tracker.append(str(l) + ',' + str(i) + ',' + str(noise))
                    if bprop:
                        self._batch._batch_optimize(self, batch_loss)

                    if save_path:
                        result = self._fprop(data)  # check for smooth fprop
                        if t.isnan(result['link probability']).max().data.item() == 0:
                            if last_loss == []:
                                with open(save_path, 'wb') as file_object:  # save
                                    t.save(obj=self, f=file_object)
                                with open(save_path + '_loss.csv', 'w') as file_object:  # save
                                    file_object.write('\n'.join(loss_tracker))
                                print('saved')
                                last_loss = l

                            elif l < last_loss:
                                with open(save_path, 'wb') as file_object:  # save
                                    t.save(obj=self, f=file_object)
                                with open(save_path + '_loss.csv', 'w') as file_object:  # save
                                    file_object.write('\n'.join(loss_tracker))
                                print('saved')
                                last_loss = l
                except:
                    print('Retrying training loop....')
        # self._train_p_map(data=data)
        return

    def get_batch_data_by_rnd_label(self, data, batch_size, mode='single'):
        # mode = single/multi labels
        t.manual_seed(time.time())
        pos_batch = []
        if mode == 'single':
            rnd_label = t.randint(self.max_label_count, (1, 1)).squeeze().data.item()
            while (len(pos_batch) == 0):
                pos_batch = self._batch._create_training_batch_pos_for_label(label=rnd_label, data=data,
                                                                             batch_size=batch_size)
                print('Retrying batch...')
            print('Current training label:', rnd_label)
        elif mode == 'multi':
            label_pos = t.randint(len(data[0]), (batch_size, 1)).squeeze()
            if label_pos.size() == ():
                label_pos = label_pos.unsqueeze(0)
            pos_batch = [[data[0][:x + 1], data[1][:x + 1], data[2][:x + 1], data[3][:x + 1]] for x in label_pos]
        batch_output = [self._fprop(x) for x in pos_batch]

        return batch_output

    def get_batch_loss(self, batch_data):
        batch_losses = [self._get_loss(d) for d in batch_data]
        return np.median([x.data.item() for x in batch_losses]), batch_losses

    def get_neg_batch_loss(self, batch_data, noise):
        neg_result = [self.NegativeSampleGenerator.get_random_noise_function(self=self, data=x, noise=noise) for x in
                      batch_data]
        neg_loss = [self._get_loss(x[0], truth=1) for x in neg_result]
        return np.median([x.data.item() for x in neg_loss]), neg_loss

    def _use_est(self, e, i, act=None):
        if act == None:
            act = lambda x: x
        o = act(e[0](i))
        for ee in e[1:]:
            o = act(ee(o))
        return o

    def _process_etc(self, etc_data):
        self.etc_params = {}
        etc_learning_params = []
        for i, v in enumerate(etc_data):
            for j, vv in enumerate(v):
                if isinstance(vv, list):
                    # this means that this field used to be a string,
                    # we will use lstm's cell state for the compressed version of the data
                    if i == 0:
                        self.etc_params[j] = mlt.create_lstm(input_size=1, output_size=1, batch_size=1, num_of_layers=1,
                                                             device=self.device)
                        etc_learning_params += list(self.etc_params[j][0].parameters())
                    etc_data[i][j] = t.tensor(etc_data[i][j], dtype=t.float32, device=self.device)
                else:
                    etc_data[i][j] = t.tensor(etc_data[i][j], dtype=t.float32, device=self.device).unsqueeze(0)
            if i == 0:
                self.learning_params.append({'params': etc_learning_params})
        return etc_data

    def _fprop(self, current_data):
        labels, timings, par_pos, etc = current_data
        current_time = timings[-1]
        current_label = labels[-1]
        parent_pos = current_data[2][-1]

        # check for orphans
        if parent_pos >= len(labels):
            raise Exception('parent_pos >= len(labels)')
        parent_label = labels[parent_pos]

        norm_timings = t.tanh(timings.sub(current_time).mul(self.time_scale))

        # fprop etc
        etc_f = []
        for i, v in enumerate(etc):
            tv = []
            # if i % 100 == 0:
            # print('Processing etc:', str(i), ' of ', str(len(etc)))
            if v == []:
                break

            for j, vv in enumerate(v):
                if vv.shape[0] > 1:
                    vv = vv.reshape([-1, 1, 1])
                    _, (next_hidden, compressed) = self.etc_params[j][0](vv, self.etc_params[j][1])  # very slow
                    tv.append(compressed.squeeze().unsqueeze(0))
                else:
                    tv.append(vv)
            etc_f.append(t.cat(tv).unsqueeze(0))

        label_emb = t.index_select(self.w_node_emb, dim=0, index=labels)
        if etc_f != []:
            etc_f = t.cat(etc_f, dim=0)
            emb_time_etc = t.cat([label_emb, norm_timings.unsqueeze(1), etc_f], dim=1).unsqueeze(1)
        else:
            emb_time_etc = t.cat([label_emb, norm_timings.unsqueeze(1)], dim=1).unsqueeze(1)

        c_hs = self.lstm_model(emb_time_etc, self.lstm_h_init)[1]  # h,c
        p_hs = self.lstm_model(emb_time_etc[:parent_pos + 1], self.lstm_h_init)[1]

        # current_emb = t.index_select(label_emb, 0, current_label).squeeze()
        # parent_emb = t.index_select(label_emb, 0, parent_label).squeeze()

        _s = lambda x: x.reshape(self.lstm_layers * self.embedding_dim).squeeze()
        lstm_out = t.cat([label_emb[parent_pos], _s(p_hs[1]), label_emb[-1], _s(c_hs[1])], dim=0)
        # lstm_out = t.cat([_s(p_hs[1]), _s(c_hs[1]), current_emb, parent_emb], dim=0)
        # lstm_out = t.cat([current_emb, parent_emb], dim=0)
        link = self._e_prop(self.e_lstm_out_to_h0, lstm_out, a=None)
        link = self._e_prop(self.e_h0_to_p, link, a=t.tanh).softmax(0)
        # link = link.div(link.sum())
        return {'link probability': link, 'parent': {'pos': parent_pos, 'label': labels[parent_pos]},
                'current': {'pos': len(labels), 'label': current_label}, 'data': current_data}

    def _get_loss(self, d, truth=0):
        ground_truth = t.tensor([1.0, 0.0], device=self.device) if truth == 0 else t.tensor([0.0, 1.0],
                                                                                            device=self.device)
        p_label = d['parent']['label']
        p_pos = d['parent']['pos']
        current_data = d['data']
        current_label = d['data'][0][-1]

        link_p = d['link probability']
        # print('Label:', current_label, ' Pos:', len(d['data'][0]))
        loss = t.nn.BCELoss()(link_p, ground_truth)
        return loss  # ground_truth.sub(link_p).pow(2).mean()  # loss  # link_p.sub(1).pow(2)

    class NegativeSampleGenerator:
        @staticmethod
        def clone_data(data):
            new_data = []
            for d in data:
                if not isinstance(d, list):
                    new_data.append(d.clone())
                else:
                    new_data.append([[x.clone() for x in l] for l in d])
            return new_data

        @staticmethod
        def nonparent(self, data):
            t.manual_seed(time.time())
            data = self.NegativeSampleGenerator.clone_data(data)
            label = data[0][-1]
            label_parent_pos = data[2][-1]
            nonparents = self._get_nonparents(data, label)
            rnd_nonparent = nonparents[t.randperm(len(nonparents))][0]  # shuffle

            data[0][label_parent_pos] = rnd_nonparent
            neg_out = self._fprop(current_data=data)
            return neg_out, rnd_nonparent

        @staticmethod
        def get_random_noise_function(self, data, noise):
            fn_set = [
                self.NegativeSampleGenerator.nonparent,
                self.NegativeSampleGenerator.point_mutate,
                self.NegativeSampleGenerator.point_shift
            ]
            choice = np.random.randint(len(fn_set))

            if choice == 0:
                return fn_set[choice](self=self, data=data)
            elif choice == 1 or choice == 2:
                return fn_set[choice](self=self, data=data, noise_percent=noise)
            return

        @staticmethod
        def point_mutate(self, data, scramble=['parent', 'current'], noise_percent=0.5):
            t.manual_seed(time.time())
            data = self.NegativeSampleGenerator.clone_data(data)
            label = data[0][-1]
            label_parent_pos = data[2][-1]
            label_parent = data[0][label_parent_pos]

            # scramble parent history
            if scramble.__contains__('parent'):
                for i in range(label_parent_pos):
                    if np.random.rand() < noise_percent:
                        data[0][i] = t.randint(self.max_label_count, (1, 1))[0][0]

            # scramble current history
            if scramble.__contains__('current'):
                for i in range(label_parent_pos, len(data[0])):
                    if np.random.rand() < noise_percent:
                        data[0][i] = t.randint(self.max_label_count, (1, 1))[0][0]
            neg_out = self._fprop(current_data=data)
            return neg_out, data

        @staticmethod
        def point_shift(self, data, scramble=['parent', 'current'], noise_percent=0.5,
                        random_insert_bracket_range=[10, 20]):
            old_seq, old_time, old_parent = data[0], data[1], data[2]
            data = self.NegativeSampleGenerator.clone_data(data)
            label, label_parent_pos = data[0][-1], data[2][-1]
            label_parent = data[0][label_parent_pos]

            def _h(ii):
                t.manual_seed(time.time())
                if np.random.rand() < noise_percent:
                    future = t.arange(min(i + random_insert_bracket_range[0], len(data[0]) - 1),
                                      min(i + random_insert_bracket_range[1], len(data[0]) - 1))

                    past = t.arange(max(i - random_insert_bracket_range[1], 0),
                                    max(i - random_insert_bracket_range[0], 0))

                    possible_positions = t.cat([future, past])
                    chosen_position = possible_positions[t.randperm(len(possible_positions))][0]

                    data[0][chosen_position] = old_seq[ii]
                    data[1][chosen_position] = (old_time[chosen_position - 1].add(old_time[chosen_position + 1])).div(2)
                    data[2][chosen_position] = old_parent[ii]

                    if chosen_position < ii:
                        for j in range(chosen_position + 1, ii):
                            # shift the rest to future
                            data[0][j] = old_seq[j - 1]
                            data[1][j] = old_time[j - 1]
                            data[2][j] = old_parent[j - 1]

                    else:  # move to future
                        for j in range(ii, chosen_position):
                            # shift the rest to past
                            data[0][j] = old_seq[j + 1]
                            data[1][j] = old_time[j + 1]
                            data[2][j] = old_parent[j + 1]
                return

            def _f():
                old_seq = data[0]
                old_time = data[1]
                old_parent = data[2]
                return

            # scramble parent history
            if scramble.__contains__('parent'):
                for i in range(label_parent_pos):
                    _h(i)
                    _f()

            # scramble current history
            if scramble.__contains__('current'):
                for i in range(label_parent_pos, len(data[0])):
                    _h(i)
                    _f()
            neg_out = self._fprop(current_data=data)
            return neg_out, data

        @staticmethod
        def add_etc_noise(self, data, noise_percent=0.5):
            data = self.NegativeSampleGenerator.clone_data(data)

            def _f(v):
                np.random.seed(int(time.time()))
                if np.random.rand() < noise_percent:
                    if v.numpy() < 1:
                        v = v.add(np.random.rand() - 0.5)
                    else:
                        k = 1 + np.random.rand() - 0.5
                        v = v.mul(k)
                return v

            etc = data[3]
            etc_f = [[[_f(y) for y in x] for x in v] for v in etc]

            return

    def _get_parents(self, data, label):
        all_label_pos = (data[0] == label).nonzero().squeeze()
        all_label_parent_pos = data[2].index_select(0, all_label_pos)
        all_label_parents = data[0].index_select(0, all_label_parent_pos)
        all_label_parents = all_label_parents.unique(sorted=True)
        return all_label_parents

    def _get_nonparents(self, data, label):
        parents = self._get_parents(data, label)
        all_labels = self.all_labels

        nonparents = t.ones_like(all_labels, dtype=t.uint8, device=self.device)
        for p in parents:
            nonparents = (all_labels != p).mul(nonparents)
        nonparents = all_labels.masked_select(nonparents)
        return nonparents

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
        def _create_training_batch_pos_for_label(data, label, batch_size, min_length=0):
            seq_class_data = data[0]
            label_pos = (seq_class_data == label).nonzero()

            # filters out those that are too short
            pos_filter = label_pos - min_length + 1 > 0
            label_pos = t.masked_select(label_pos, pos_filter)

            # shuffle
            label_pos = label_pos[t.randperm(label_pos.shape[0])]
            if label_pos.shape[0] > batch_size:
                label_pos = label_pos[:batch_size]
            batch_data = [[data[0][:x + 1], data[1][:x + 1], data[2][:x + 1], data[3][:x + 1]] for x in label_pos]
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
