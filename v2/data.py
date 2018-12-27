import numpy as np
import time
import os
import pickle


class data_generator():
    def __init__(self):
        return

    def generate_data_from_saved_state(self, saved_path, seq_len, reuse_local=True, reuse_remote=True):
        seq_data = np.zeros(seq_len, dtype=np.int64)
        seq_time = np.zeros(seq_len, dtype=np.float32)

        if os.path.exists(saved_path):
            with open(saved_path, 'rb') as f:
                self = pickle.load(f)
        else:
            print('Load error')
            return

        if not reuse_local:
            self.p_mat = None
        if not reuse_remote:
            self.adj_mat = None
        seq_data, seq_time = self.effects.apply_local_neighbour_effect(self=self, seq_data=seq_data, seq_time=seq_time)
        seq_data, seq_time = self.effects.apply_remote_neighbour_effect(self=self, seq_data=seq_data,
                                                                        seq_time_data=seq_time)
        return seq_data, seq_time

    def generate_data(self, seq_len, class_count, save_path=None, load_path=None):
        np.random.seed(seed=int(time.time()))
        seq_data_time = [[np.random.randint(0, class_count), 0]]

        if load_path:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    self = pickle.load(f)

        try:
            p_adj_map = self.adj_map
            time_map = self.time_map
        except:

            p_adj_map = np.clip(np.power(np.multiply(np.random.uniform(size=(class_count, class_count)), 1.2), 3.3),
                                a_min=0, a_max=1)

            time_map = np.multiply(np.power(np.abs(np.random.normal(size=(class_count, class_count), scale=0.5)), 2.3),
                                   seq_len)

            self.adj_map = p_adj_map
            self.time_map = time_map

        for i in range(seq_len):

            last_label = seq_data_time[i][0]
            last_label_adj_map = p_adj_map[last_label]
            last_label_time_map = time_map[last_label]

            temp_next_labels = []
            for n in range(len(last_label_adj_map)):
                rng = np.random.rand()
                if rng < last_label_adj_map[n]:
                    temp_next_labels.append([n, last_label_time_map[n]])

            if len(temp_next_labels) > 0:
                temp_next_labels = sorted(temp_next_labels, key=lambda x: x[1])
                current_time = seq_data_time[i][1]
                for nl in temp_next_labels:
                    seq_data_time.append([nl[0], current_time + nl[1]])

        seq_data_time.sort(key=lambda x: x[1])

        seq_data = [x[0] for x in seq_data_time]
        seq_time = [x[1] for x in seq_data_time]

        if save_path:
            with open(save_path, 'wb') as save_obj:
                pickle.dump(obj=self, file=save_obj)

        return seq_data[:seq_len], seq_time[:seq_len]

    class effects():
        @staticmethod
        def apply_local_neighbour_effect(self, seq_data, seq_time, time_scalar=5, neighbour_effect=5,
                                         time_neighbour_effect=False):
            # USE A BLANK ARRAY!!!!
            # neighbour effect, number of previous neighbours that will influence the p of current label
            # time_neighbour_effect, True if you want current label time to get affected by neighbours as well

            seq_len = len(seq_data)
            class_count = self.class_count
            labels = np.arange(class_count)

            try:
                p_mat = self.p_mat
                p_time_mat = self.p_time_mat
                if len(p_mat) < 0:
                    raise Exception()
            except:
                print('Using new p_mat...')
                p_mat = data_generator.map_generator.generate_prob_adj_map(class_count)
                p_time_mat = data_generator.map_generator.generate_num_map(class_count, time_scalar)

                self.p_mat = p_mat
                self.p_time_mat = p_time_mat

            if np.all(seq_data == 0):
                seq_data[0] = np.random.randint(0, class_count)
            for i in range(1, seq_len):
                last_label = seq_data[i - 1]
                label_prob = p_mat[last_label]
                # if previous label also affect the prob of current label
                for j in range(np.min([i, neighbour_effect]), 0, -1):
                    prev_label = seq_data[j - 1]
                    prev_label_prob = p_mat[prev_label]

                    # Create neighbouring influence function here
                    label_prob = label_prob + 0.2 * j * prev_label_prob  # np.multiply(label_prob, prev_label_prob)

                # normalization
                d = np.sum(label_prob)
                label_prob = label_prob / d
                new_label = np.random.choice(labels, p=label_prob)
                seq_data[i] = new_label

                # once new label is known, create a time for it to happen
                new_label_time = p_time_mat[seq_data[-1], new_label]

                if time_neighbour_effect:
                    # create neighbour effect for time
                    for j in range(np.min([i, neighbour_effect]), 0, -1):
                        prev_label_time = seq_time[j - 1]
                        # design how this new label's time is affected by previous neighbours
                        new_label_time = 0.1 * j * np.power(prev_label_time, 8) + new_label_time

                seq_time[i] = new_label_time

            # change time values to accumulated one
            for i in range(1, len(seq_time)):
                seq_time[i] = seq_time[i - 1] + seq_time[i]

            return seq_data, seq_time

        @staticmethod
        def apply_remote_neighbour_effect(self, seq_data, seq_time_data, time_scalar=0):
            # adj_time_mat=None, time_scalar=None):
            # USE ON A FILLED ARRAY!
            seq_len = len(seq_data)
            class_count = self.class_count
            if time_scalar == 0:
                time_scalar = seq_len

            try:
                adj_mat = self.adj_mat
                adj_time_mat = self.adj_time_mat
                if len(adj_mat) < 0:
                    raise Exception()
            except:
                print('Using new adj_mat...')
                adj_mat = data_generator.map_generator.generate_adj_map(class_count, offset=0.4)
                adj_time_mat = time_scalar * data_generator.map_generator.generate_prob_adj_map(class_count)

                self.adj_mat = adj_mat
                self.adj_time_mat = adj_time_mat

            i = 0
            while True:
                current_label = seq_data[i]
                current_label_adj = adj_mat[current_label]
                next_label_time_mat = adj_time_mat[current_label]

                next_labels_pos = np.argwhere(current_label_adj == 1)
                for p in next_labels_pos:
                    next_label = p
                    next_label_increment_time = next_label_time_mat[p]

                    current_time = seq_time_data[i]
                    new_label_time = current_time + next_label_increment_time
                    try:
                        insert_pos = np.where(seq_time_data > new_label_time)[0][0]
                        if insert_pos < seq_len:
                            seq_data = np.insert(seq_data, insert_pos, next_label)
                            seq_time_data = np.insert(seq_time_data, insert_pos, new_label_time)
                    except:
                        # pos too far...
                        pass

                if i == seq_len:
                    break
                i += 1
            return seq_data, seq_time_data

    class map_generator():
        @staticmethod
        def generate_prob_adj_map(class_count):
            np.random.seed(seed=int(time.time()))
            p_map = np.random.uniform(size=(class_count, class_count))
            return p_map

        @staticmethod
        def generate_num_map(class_count, scale):  # for time
            np.random.seed(seed=int(time.time()))
            p_num_map = scale * np.random.uniform(size=(class_count, class_count))
            return p_num_map

        @staticmethod
        def generate_adj_map(class_count, offset):
            adj_map = data_generator.map_generator.generate_prob_adj_map(class_count)
            adj_map = np.subtract(adj_map, offset)
            adj_map = np.round(adj_map)
            return adj_map
