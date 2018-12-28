import numpy as np
import time
import os
import pickle


class data_generator():
    def __init__(self):
        return

    def generate_data(self, seq_len, class_count, save_path=None, load_path=None):
        np.random.seed(seed=int(time.time()))

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

        # we start the sequence with the class that has the most prob of generating other classes
        seed_class = np.argmax(np.sum(p_adj_map, axis=1))
        seq_data_time = [[seed_class, 0, 0]]

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
                    next_time = current_time + nl[1]
                    if next_time >= seq_data_time[-1][1]:
                        seq_data_time.append([nl[0], current_time + nl[1], i])
                    else:
                        for j in range(len(seq_data_time) - 1, 0, -1):
                            old_time = seq_data_time[j][1]
                            if old_time <= next_time:
                                seq_data_time.insert(j + 1, [nl[0], current_time + nl[1], i])
                                break

        seq_data_time.sort(key=lambda x: x[1])

        seq_data = np.asarray([x[0] for x in seq_data_time][:seq_len], dtype=np.int64)
        seq_time = np.asarray([x[1] for x in seq_data_time][:seq_len], dtype=np.float32)
        seq_adj = np.asarray([x[2] for x in seq_data_time][:seq_len], dtype=np.int64)

        if save_path:
            with open(save_path, 'wb') as save_obj:
                pickle.dump(obj=self, file=save_obj)

        return seq_data, seq_time, seq_adj

    @staticmethod
    def add_orphan_data(self):
        return
