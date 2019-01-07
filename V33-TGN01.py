import torch as t
from data import *
import pickle
from ml_helper.ml_helper import torch_helpers as mlt
import os
import time

data_folder = '/'.join(os.getcwd().split('/')[:-1] + ['data'])

tf = t.float
ti = t.int
device = t.device("cpu")

embedding_dim = 13
seq_len = 700

# create classes
max_class_count = 8
min_length = 50

v = lambda x: x.data.item()


class v33tgn01():
    def __init__(self, max_label_count, window_size, lstm_layers=1, embedding_dim=4, time_scale=0.01,
                 min_length=0):
        self.max_label_count = max_label_count
        self.all_labels = t.arange(0, self.max_label_count, device=device)
        self.one_hot = t.eye(max_label_count, device=device)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.time_scale = time_scale
        self.lstm_layers = lstm_layers
        self.min_length = min_length

        return

    def create_model(self):

        # weights
        self.w_node_emb = t.randn((self.max_label_count, embedding_dim), dtype=tf, device=device, requires_grad=True)

        # batch size 1 for ease of usability
        self.lstm_model, self.lstm_h_init = mlt.create_lstm(input_size=self.embedding_dim + 1,  # + x for each param
                                                            output_size=self.embedding_dim, batch_size=1,
                                                            num_of_layers=self.lstm_layers, device=device)
        # self.lstm_layers * embedding_dim * 2 + embedding_dim * 2
        self.e_lstm_out_to_h0, e1 = mlt.create_linear_layers(
            layer_sizes=[self.lstm_layers * embedding_dim * 4, 30, embedding_dim * 2], device=device)
        self.e_h0_to_p, e2 = mlt.create_linear_layers(layer_sizes=[embedding_dim * 2, 1], device=device)

        # learning_weights = [self.w_node_emb] + e1 + e2 + e3 + e4 + e5 + e6 + list(self.lstm_model.parameters())
        learning_weights = [self.w_node_emb] + e1 + e2 + list(self.lstm_model.parameters())

        self.learning_params = [
            {'params': learning_weights},
            # {'params': self.lstm_model.parameters()}
        ]
        return

    def train_model(self, data, batch_size=10, iter=100, lr=1e-3, save_path=None, bprop=True, neg_sample_cnt=10):
        _f = lambda x: t.tensor(x, device=device)
        data = [_f(x) for x in data]
        try:  # sorry, shouldnt be doing this way but its convenient
            self.w_node_emb
        except:
            self.create_model()

        self.optim = t.optim.SGD(self.learning_params, lr=lr)

        i = 0
        for i in range(iter):
            batch_data = self.get_batch_data_by_rnd_label(data, batch_size)
            l, batch_loss = self.get_batch_loss(batch_data=batch_data, neg_sample_cnt=neg_sample_cnt)

            print('Batch loss:', l)
            if bprop:
                self._batch._batch_optimize(self, batch_loss)

            i += 1

            if save_path:
                if i % 10 == 0 and i != 0:
                    link_prop = self._fprop(data)['link probability']
                    # checking for NaNs
                    if t.isnan(link_prop).max().data.item() == 1:
                        raise Exception('NaN for training params, skipped saving model...')
                    with open(save_path, 'wb') as file_object:  # save
                        t.save(obj=self, f=file_object)
                        # pickle.dump(obj=self, file=file_object)
                        print('saved')
                    i = 0
        return

    def get_batch_data_by_rnd_label(self, data, batch_size):

        t.manual_seed(time.time())
        rnd_label = t.randint(self.max_label_count, (1, 1)).squeeze().data.item()
        pos_batch = []
        while (len(pos_batch) == 0):
            pos_batch = self._batch._create_training_batch_pos_for_label(label=rnd_label, data=data,
                                                                         batch_size=batch_size)
            print('Retrying batch...')
        print('Current training label:', rnd_label)
        batch_output = [self._fprop(x) for x in pos_batch]

        return batch_output

    def get_batch_loss(self, batch_data, neg_sample_cnt=5):
        batch_losses = [self._get_loss(d, neg_sample_cnt=neg_sample_cnt) for d in batch_data]
        return np.mean([x.data.item() for x in batch_losses]), batch_losses

    def _use_est(self, e, i, act=None):
        if act == None:
            act = lambda x: x
        o = act(e[0](i))
        for ee in e[1:]:
            o = act(ee(o))
        return o

    def _fprop(self, current_data):
        # check for orphans
        labels, timings, par_pos = current_data
        current_time = timings[-1]
        current_label = labels[-1]
        parent_pos = current_data[2][-1]

        # check for orphans
        parent_label = labels[parent_pos]

        norm_timings = t.tanh(timings.sub(current_time).mul(self.time_scale))

        label_emb = t.index_select(self.w_node_emb, dim=0, index=labels)
        emb_time = t.cat([label_emb, norm_timings.unsqueeze(1)], dim=1).unsqueeze(1)

        c_hs = self.lstm_model(emb_time, self.lstm_h_init)[1]
        p_hs = self.lstm_model(emb_time[:parent_pos + 1], self.lstm_h_init)[1]

        # current_emb = t.index_select(label_emb, 0, current_label).squeeze()
        # parent_emb = t.index_select(label_emb, 0, parent_label).squeeze()

        _s = lambda x: x.reshape(self.lstm_layers * embedding_dim).squeeze()
        lstm_out = t.cat([_s(p_hs[0]), _s(p_hs[1]), _s(c_hs[0]), _s(c_hs[1])], dim=0)
        # lstm_out = t.cat([_s(p_hs[1]), _s(c_hs[1]), current_emb, parent_emb], dim=0)
        # lstm_out = t.cat([current_emb, parent_emb], dim=0)
        link = self._e_prop(self.e_lstm_out_to_h0, lstm_out, a=None)
        link = self._e_prop(self.e_h0_to_p, link, a=t.sigmoid)
        return {'link probability': link, 'parent': {'pos': parent_pos, 'label': labels[parent_pos]},
                'current': {'pos': len(labels), 'label': current_label}, 'data': current_data}

    def _fprop_a(self, current_data):
        # designed as such for non parents
        labels, timings, par_pos = current_data
        current_time = timings[-1]
        current_label = labels[-1]
        parent_pos = current_data[2][-1]

        # check for orphans
        parent_label = labels[parent_pos]

        norm_timings = t.tanh(timings.sub(current_time).mul(self.time_scale))

        label_emb = t.index_select(self.w_node_emb, dim=0, index=labels)
        emb_time = t.cat([label_emb, norm_timings.unsqueeze(1)], dim=1).unsqueeze(1)

        c_hs = self.lstm_model(emb_time, self.lstm_h_init)[1]
        p_hs = self.lstm_model(emb_time[:parent_pos + 1], self.lstm_h_init)[1]

        # current_emb = t.index_select(label_emb, 0, current_label).squeeze()
        # parent_emb = t.index_select(label_emb, 0, parent_label).squeeze()

        _s = lambda x: x.reshape(self.lstm_layers * embedding_dim).squeeze()
        lstm_out = t.cat([_s(p_hs[0]), _s(p_hs[1]), _s(c_hs[0]), _s(c_hs[1])], dim=0)
        # lstm_out = t.cat([_s(p_hs[1]), _s(c_hs[1]), current_emb, parent_emb], dim=0)
        # lstm_out = t.cat([current_emb, parent_emb], dim=0)
        link = self._e_prop(self.e_lstm_out_to_h0, lstm_out, a=None)
        link = self._e_prop(self.e_h0_to_p, link, a=t.sigmoid)
        return {'link probability': link, 'parent': {'pos': parent_pos, 'label': labels[parent_pos]},
                'current': {'pos': len(labels), 'label': current_label}, 'data': current_data}

    def _get_loss(self, d, neg_sample_cnt=3):
        p_label = d['parent']['label']
        p_pos = d['parent']['pos']
        current_data = d['data']
        current_label = d['data'][0][-1]

        link_p = d['link probability']
        bce_gtruth = t.ones(1, device=device)

        for n in range(neg_sample_cnt):
            t.manual_seed(time.time())
            # select non parents
            neg = [
                # self.NegativeSampleGenerator.add_etc_noise(self, data=d['data'], noise_percent=0.3)[0],
                self.NegativeSampleGenerator.nonparent(self, data=d['data'])[0],
                self.NegativeSampleGenerator.point_mutate(self, data=d['data'], noise_percent=0.3)[0],
                self.NegativeSampleGenerator.point_shift(self, data=d['data'], noise_percent=0.3)[0]

            ]
            link_p = t.cat([link_p] + [x['link probability'] for x in neg], dim=0)
            bce_gtruth = t.cat([bce_gtruth] + [t.zeros(1, device=device) for x in neg])

        link_p = link_p.div(link_p.sum())
        print(link_p)
        bce_loss = t.nn.BCELoss()(link_p, bce_gtruth)
        return bce_loss  # bce_gtruth.sub(link_p).pow(2).mean()  # bce_loss  # link_p.sub(1).pow(2)

    class NegativeSampleGenerator:
        @staticmethod
        def clone_data(data):
            new_data = []
            for d in data:
                new_data.append(d.clone())
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

            # scramble parent history
            if scramble.__contains__('parent'):
                for i in range(label_parent_pos):
                    _h(i)

            # scramble current history
            if scramble.__contains__('current'):
                for i in range(label_parent_pos, len(data[0])):
                    _h(i)
            neg_out = self._fprop(current_data=data)
            return neg_out, data

    def _get_parents(self, data, label):
        all_label_pos = (data[0] == label).nonzero().squeeze()
        all_label_parent_pos = data[2].index_select(0, all_label_pos)
        all_label_parents = data[0].index_select(0, all_label_parent_pos)
        all_label_parents = all_label_parents.unique(sorted=True)
        return all_label_parents

    def _get_nonparents(self, data, label):
        parents = self._get_parents(data, label)
        all_labels = self.all_labels

        nonparents = t.ones_like(all_labels, dtype=t.uint8, device=device)
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
        def _create_training_batch_pos_for_label(data, label, batch_size):
            seq_class_data = data[0]
            label_pos = (seq_class_data == label).nonzero()

            # filters out those that are too short
            pos_filter = label_pos - min_length + 1 > 0
            label_pos = t.masked_select(label_pos, pos_filter)

            # shuffle
            label_pos = label_pos[t.randperm(label_pos.shape[0])]
            if label_pos.shape[0] > batch_size:
                label_pos = label_pos[:batch_size]
            batch_data = [[data[0][:x + 1], data[1][:x + 1], data[2][:x + 1]] for x in label_pos]
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


if __name__ == '__main__':
    data_map_path = data_folder + '/dataV4'
    save_path = data_folder + '/modelV4'

    save_mode = True
    load_from_old = False

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

    model = v33tgn01(time_scale=0.001, max_label_count=max_class_count,
                     window_size=min_length, embedding_dim=embedding_dim, lstm_layers=3)
    if load_from_old:
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file_object:  # load
                model = t.load(file_object, map_location=device)

    model.train_model(data, save_path=save_path, batch_size=10, lr=1e-2, neg_sample_cnt=1)
