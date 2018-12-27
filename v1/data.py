import numpy as np
import time
import os
import pickle

def generate_data(max_class_count, seq_len, load_path=None, save_path=None):
    labels = np.arange(max_class_count, dtype=np.int32)

    if load_path:
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print('Load error')
            return

    # generate 2D probability diagram
    # if you want to use edges, it will be an adjacency matrix
    # or you can mix both
    if not load_path:
        np.random.seed(seed=int(time.time()))
        p_map = np.random.uniform(size=(max_class_count, max_class_count))
        p_time_map = np.random.uniform(size=(max_class_count, max_class_count))
    else:
        p_map = data[0]
        p_time_map = data[1]

    # generate sample sequential data
    seq_data = [np.random.randint(0, max_class_count)]
    seq_time_data = [0]

    # local neighbour effect
    neighbour_effect = 3

    for i in range(seq_len):
        last_label = seq_data[-1]
        label_prob = p_map[last_label]
        # if previous label also affect the prob of current label
        for j in range(np.min([i, neighbour_effect]), 0, -1):
            prev_label = seq_data[j - 1]
            prev_label_prob = p_map[prev_label]

            # Create neighbouring influence function here
            label_prob = label_prob + 0.2 * j * prev_label_prob  # np.multiply(label_prob, prev_label_prob)

        # normalization
        d = np.sum(label_prob)
        label_prob = label_prob / d

        new_label = np.random.choice(labels, p=label_prob)

        seq_data.append(new_label)

        # once new label is known, create a time for it to happen
        new_label_time = p_time_map[seq_data[-1], new_label]
        # create neighbour effect for time
        for j in range(np.min([i, neighbour_effect]), 0, -1):
            prev_label = seq_data[j - 1]
            prev_label_time = seq_time_data[j - 1]

            # design how this new label's time is affected by previous neighbours
            new_label_time = 0.1 * j * np.power(prev_label_time, 8) + new_label_time

        seq_time_data.append(new_label_time)

    # change time values to accumulated one
    for i in range(1, len(seq_time_data)):
        seq_time_data[i] = seq_time_data[i - 1] + seq_time_data[i]

    if not load_path:
        # layering remote effects to current data using adjacency matrix
        adjacency_matrix = np.round(np.random.uniform(size=(max_class_count, max_class_count)))
        adjacency_time_matrix = 0.3 * seq_len * np.random.uniform(size=(max_class_count, max_class_count))
    else:
        adjacency_matrix = data[2]
        adjacency_time_matrix = data[3]

    i = 0
    while True:
        current_label = seq_data[i]
        current_label_adjacency = adjacency_matrix[current_label]
        next_label_time_mat = adjacency_time_matrix[current_label]

        next_labels_pos = np.argwhere(current_label_adjacency == 1)
        for p in next_labels_pos:
            next_label = p
            next_label_increment_time = next_label_time_mat[p]

            current_time = seq_time_data[i]
            new_label_time = current_time + next_label_increment_time
            try:
                insert_pos = np.where(seq_time_data > new_label_time)[0][0]
                seq_data = np.insert(seq_data, insert_pos, next_label)
                seq_time_data = np.insert(seq_time_data, insert_pos, new_label_time)
            except:
                pass

        if i == seq_len:
            break
        i += 1
    if save_path:
        with open(save_path, 'wb') as file_object:  # save
            pickle.dump(obj=[p_map, p_time_map, adjacency_matrix, adjacency_time_matrix], file=file_object)
    seq_data = seq_data.astype(np.int64)
    seq_time_data = seq_time_data.astype(np.float32)
    return seq_data[:seq_len], seq_time_data[:seq_len]
