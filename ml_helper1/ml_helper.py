import tensorflow as tf
import torch as t
import numpy as np

class SimpleNeuralNetworkModel:
    # for instant usage
    @staticmethod
    def create_simple_autoencoder(x_input):
        from keras import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(units=100, input_shape=(x_input.shape[1],)))
        model.add(Dense(units=10, activation='sigmoid'))
        model.add(Dense(units=100))
        model.add(Dense(x_input.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        return model

class DataGenerators:
    class Unsupervised:
        @staticmethod
        def transform_1D_data_to_reverse_dist(data, new_sample_ratio=False, return_same_sized_combined_dist=True,
                                              bins=30,
                                              imba_f=1.2,
                                              visualization=True):
            from keras import Input, Sequential, Model
            from keras.layers import Dense
            from keras.optimizers import Adam
            from keras.callbacks import EarlyStopping
            import matplotlib.pyplot as plt
            # instead of making rare events having the same standing as frequent events, we make rare events even more common than norm
            # imba factor controls the distribution of rare events > normal events

            # if no_of_new_samples is not specified, it attempts to calculate the number by finding the amount of new samples
            # required to fill up the remaining area of the uniform dist (think of it as the unfilled area of a rectangle'
            if new_sample_ratio == 0 and new_sample_ratio != False or imba_f == 0:
                return data

            latent_dim = 1
            feature_count = len(data[0])
            enc_input = Input(shape=(feature_count,))

            encoder = Sequential()
            encoder.add(Dense(100, input_shape=(feature_count,)))
            encoder.add(Dense(latent_dim))

            decoder = Sequential()
            decoder.add(Dense(100, input_shape=(latent_dim,)))
            decoder.add(Dense(feature_count))

            final = Model(enc_input, decoder(encoder(enc_input)))
            final.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")

            np.random.shuffle(data)

            final.fit(x=np.asarray(data), y=np.asarray(data), batch_size=int(len(data) / 10),
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0.00001)],
                      epochs=500)

            latent_values = encoder.predict(data)

            if visualization:
                plt.figure('Original latent values histogram')
                plt.hist(latent_values, bins=bins)

            if bins > len(latent_values):
                bins = int(len(latent_values) / 2)
            count, ranges = np.histogram(latent_values, bins=bins)

            no_of_new_samples = 0
            if not new_sample_ratio:
                no_of_new_samples = np.sum(np.max(count) - count)
            else:
                no_of_new_samples = int(len(data) * new_sample_ratio)

            bins_probability_table = [np.power(x, imba_f) for x in np.rint(max(count) - count) / max(count)]
            bins_probability_table /= np.max(bins_probability_table)

            new_latent_values = []

            while (True):
                for i in range(len(bins_probability_table)):
                    bin_rng = [ranges[i], ranges[i + 1]]
                    bins_prob = bins_probability_table[i]
                    if np.random.rand() < bins_prob:
                        new_synth_latent = np.random.rand() * (bin_rng[1] - bin_rng[0]) + bin_rng[0]
                        new_latent_values.append([new_synth_latent])
                    if len(new_latent_values) >= no_of_new_samples:
                        break
                if len(new_latent_values) >= no_of_new_samples:
                    break

            # for debugging
            if len(new_latent_values) == 0:
                return data
            new_synth_data = decoder.predict(np.asarray(new_latent_values))

            if visualization:
                plt.figure('New latent values histogram')
                plt.hist(np.asarray(new_latent_values), bins=bins)

                plt.figure('Combined latent values histogram')
                combined_latent_values = np.concatenate((np.asarray(new_latent_values), latent_values))

                plt.hist(combined_latent_values, bins=bins)
                plt.show()

            # count_, ranges_ = np.histogram(new_latent_values, bins=bins)

            if return_same_sized_combined_dist == True:
                resampled_data = np.concatenate((data, new_synth_data))
                np.random.shuffle(resampled_data)
                resampled_data = resampled_data[:len(data)]

                # for debugging
                # debugging_latent_v = encoder.predict(resampled_data)
                # plt.hist(debugging_latent_v, bins=bins)
                # plt.show()

                return resampled_data
            return new_latent_values

        def transform_1D_samples_using_DOPE(data, return_same_sized_combined_dist=True, new_sample_ratio=0.3,
                                            no_of_std=3, visualization=False):
            from keras import Input, Sequential, Model
            from keras.layers import Dense
            from keras.optimizers import Adam
            from keras.callbacks import EarlyStopping
            import matplotlib.pyplot as plt
            from scipy.stats import chi

            if new_sample_ratio == 0 or no_of_std == 0:
                return data

            latent_dim = 1
            no_of_new_samples = int(len(data) * new_sample_ratio)
            feature_count = len(data[0])
            enc_input = Input(shape=(feature_count,))

            encoder = Sequential()
            encoder.add(Dense(100, input_shape=(feature_count,)))
            encoder.add(Dense(latent_dim))

            decoder = Sequential()
            decoder.add(Dense(100, input_shape=(latent_dim,)))
            decoder.add(Dense(feature_count))

            final = Model(enc_input, decoder(encoder(enc_input)))
            final.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")

            np.random.shuffle(data)

            final.fit(x=np.asarray(data), y=np.asarray(data), batch_size=int(len(data) / 10),
                      callbacks=[EarlyStopping(monitor='loss', min_delta=0.00001)],
                      epochs=500)

            latent_values = encoder.predict(data)

            if visualization:
                # for debugging of distribution of latent_values
                plt.figure('Latent value distribution')
                plt.hist(latent_values, bins=30)
                plt.show()

            center = np.mean(latent_values, axis=0)
            std = np.std(latent_values, axis=0)
            chi_std = chi.std(2, 0, np.linalg.norm(std))

            # x-mean
            # I have a problem with the following line, he assumes that the latent values are already gaussian
            # distributed hence using it directly
            dist = np.linalg.norm(latent_values - center, axis=1)  # Frobenius norm

            if visualization:
                # for debugging of distribution
                plt.figure('L1 norm distribution')
                plt.hist(dist, bins=30)
                plt.show()

            for i, el in enumerate(dist):
                dist[i] = 0. if el > no_of_std * chi_std else dist[i]

            if visualization:
                # for debugging of distribution
                plt.figure('L1 norm distribution after std filtering')
                plt.hist(list(filter(lambda x: x > 0, dist)), bins=30)
                plt.show()

            threshold = sorted(dist)[int(len(dist) * 0.9)]  # this is cutting too much

            dist = [0. if x < threshold else x for x in dist]

            if visualization:
                # for debugging of distribution
                plt.figure('L1 norm distribution after std & threshold filtering')
                plt.hist(list(filter(lambda x: x > 0, dist)), bins=30)
                plt.show()

            dist /= np.sum(dist)

            synth_latent = []
            for i in range(no_of_new_samples):
                # choose an ele from 1st argv, given that 1st argv has prob dist in p
                choice = np.random.choice(np.arange(len(dist)), p=dist)

                a = latent_values[choice]
                latent_copy = np.concatenate((latent_values[:choice], latent_values[choice + 1:]))
                latent_copy -= a
                latent_copy = np.linalg.norm(latent_copy, axis=1)  # Frobenius norm
                b = np.argmin(latent_copy)
                if b >= choice:
                    b += 1
                b = latent_values[b]
                scale = np.random.rand()
                c = scale * (a - b) + b
                synth_latent.append(c)

            new_latent_values = np.concatenate((latent_values, np.asarray(synth_latent)))

            new_data = decoder.predict(np.asarray(synth_latent))
            if return_same_sized_combined_dist:
                resampled_data = np.concatenate((data, new_data))
                np.random.shuffle(resampled_data)
                return resampled_data[:len(data)]
            return new_data

        @staticmethod
        def __helper(x):
            from keras.callbacks import EarlyStopping
            simple_autoencoder = SimpleNeuralNetworkModel.create_simple_autoencoder(x)
            np.random.shuffle(x)
            early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001)
            history = simple_autoencoder.fit(x=x, y=x, batch_size=int(len(x) / 10), epochs=500,
                                             callbacks=[early_stopping])
            loss = Hidden.get_loss(simple_autoencoder, x=x, y=x, type='max')
            return loss


class tf_helpers():
    @staticmethod
    def create_layers(layer_sizes, name=''):
        output = []
        for i in range(len(layer_sizes) - 1):
            if not output:
                rng_seed = tf.random_normal([layer_sizes[i], layer_sizes[i + 1]], stddev=0.1)
                output.append(tf.Variable(rng_seed, name=name + '_w' + str(i)))
        if name != '':
            output = tf.identity(output, name=name)
        return output

    @staticmethod
    def matmul_activate(tf_matrix, activations=[], name=''):
        # example, activations = [[],tf.tanh,tf.sigmoid,[]]
        # eg matmul_activate(create_layers
        output = []
        for i in range(len(tf_matrix) - 1):
            input = tf_matrix[0] if i == 0 else output
            if i < len(activations):
                if activations[i]:
                    input = activations[i](input)
            output = tf.matmul(input, tf_matrix[i + 1])
        if name != '':
            output = tf.identity(output, name=name)
        return output


class torch_helpers():
    @staticmethod
    def create_linear_layers(layer_sizes, use_bias=True, device='cpu'):
        # returns linear, learning params
        l = []
        prev_size = layer_sizes[0]
        for s in layer_sizes[1:]:
            m = t.nn.Linear(in_features=prev_size, out_features=s, bias=use_bias)
            m.to(device)
            l.append(m)
            prev_size = s

        lt = []
        for ll in l:
            lt += list(ll.parameters())
        return l, lt

    @staticmethod  # enforces a strict structure
    def create_lstm(input_size, output_size, batch_size, num_of_layers, bidirectional=False, device='cpu'):
        # returns model, hidden_states for propagation
        t.manual_seed(1)
        # data details
        hidden_size = output_size

        # model details
        num_directions = 2 if bidirectional else 1

        lstm = t.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_of_layers,
                         bidirectional=bidirectional)

        # initialize the hidden state.
        hidden = (t.zeros(num_of_layers * num_directions, batch_size, hidden_size, device=device),  # this is for h_0,
                  t.zeros(num_of_layers * num_directions, batch_size, hidden_size,
                          device=device))  # this is for c_0, cell state

        # inputs = t.randn(seq_len, batch_size, input_dim)
        # out, hidden = lstm(inputs, hidden)
        lstm.to(device=device)
        return lstm, hidden

    @staticmethod
    def create_lstm_cell(input_size, output_size, batch_size, bidirectional=False, bias=True, device='cpu'):
        t.manual_seed(1)
        lstm_cell = t.nn.LSTMCell(input_size=input_size, hidden_size=output_size, bias=bias)

        # initialize the hidden state.
        hidden = (t.zeros(batch_size, output_size, device=device),  # this is for h_0,
                  t.zeros(batch_size, output_size, device=device))  # this is for c_0, cell state

        return lstm_cell, hidden

    @staticmethod
    def prod_estimator(x, y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=1000):
        # for forms Ca^m.b^n
        # can only be used for +ve numbers,
        w_coefficient = t.randn([1], dtype=dtype, device=device,
                                requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)
        w_power = t.ones(x.shape[1], y.shape[1], device=device, dtype=dtype, requires_grad=True)

        # w_power = t.tensor([[2],[1]], device=device, dtype=dtype, requires_grad=True)

        def fprop(x):
            y_ = t.log(x)
            y_ = y_.mm(w_power)  # this controls mul/divide
            y_ = t.sum(y_, dim=1)
            y_ = t.exp(y_)
            y_ = y_.mul(w_coefficient)  # accounts for coefficient
            y_ = y_.reshape([len(y_), 1])
            return y_

        loss_fn = lambda y_, y: (y_ - y).pow(2).mean()

        def bprop(x, y, w):
            w_power, w_coefficient = w[0], w[1]
            y_ = fprop(x)
            l = loss_fn(y_, y)
            l.backward()

            with t.no_grad():
                w_power -= lr * w_power.grad
                w_coefficient -= lr * w_coefficient.grad
                print(i, ' ', l.item())

                w_power.grad.zero_()
                w_coefficient.grad.zero_()
            return [w_power, w_coefficient]

        for i in range(iter):
            bprop(x, y, [w_power, w_coefficient])
        return [w_power, w_coefficient], fprop, bprop

    @staticmethod  # unfinished
    def sum_estimator(x, y, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=500000):
        # for forms Aa^x+Bb^y+Cc^z
        w_coefficient = t.ones([x.shape[1], 1], dtype=dtype, device=device,
                               requires_grad=True)
        w_power = t.ones([x.shape[1]], dtype=dtype, device=device,
                         requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)

        def fprop(x):
            y_ = t.pow(x, w_power)
            y_ = y_.mm(w_coefficient)
            return y_

        loss_fn = lambda y_, y: (y_ - y).pow(2).mean()

        def bprop(x, y, w):
            w_power, w_coefficient = w[0], w[1]
            y_ = fprop(x)
            l = loss_fn(y_, y)
            l.backward()
            with t.no_grad():
                w_power -= lr * w_power.grad
                w_coefficient -= lr * w_coefficient.grad
                print(l.item())
                w_power.grad.zero_()
                w_coefficient.grad.zero_()
            return [w_power, w_coefficient]

        for i in range(iter):
            bprop(x, y, [w_power, w_coefficient])
        return [w_power, w_coefficient], fprop, bprop

    @staticmethod
    def composite_estimator(x, y, hidden=10, device=t.device("cpu"), dtype=t.float, lr=1e-2, iter=50):

        composite_set = [
            torch_helpers.prod_estimator(x, y, iter=0),
            torch_helpers.prod_estimator(x, y, iter=0)
        ]

        w_hidden = t.randn([hidden * len(composite_set), y.shape[1]], dtype=dtype, device=device,
                           requires_grad=True)  # t.randn(1, device=device, dtype=dtype, requires_grad=True)

        set_output = [x[1](x) for x in composite_set]
        set_output = t.cat(set_output, dim=1)

        # fn_ar = [[t.sigmoid,3],[t.tanh,3]]
        return


class Hidden:
    @staticmethod
    def get_bin_pos(v, bin_rng, return_bin_range=False):
        if v < bin_rng[0]:
            return [0, [bin_rng[0], bin_rng[1]]] if return_bin_range else 0
        for i in range(len(bin_rng) - 1):
            if bin_rng[i] <= v < bin_rng[i + 1]:
                return [i, [bin_rng[i], bin_rng[i + 1]]] if return_bin_range else i
        return [i, [bin_rng[-2], bin_rng[-1]]] if return_bin_range else i

    @staticmethod
    def parse_numpy_where_results(np_where_results):
        return np.asarray(np_where_results).transpose()[0]

    @staticmethod
    def get_loss(model, x, y, type='mean'):
        # type = {'mean', 'max'}
        y_ = model.predict(np.asarray(x))
        if type == 'mean':
            return np.mean(np.abs(np.add(y_, -y)), axis=1)
        elif type == 'mean_square':
            return np.mean(np.power(np.add(y_, -y), 2), axis=1)
        elif type == 'max':
            return np.asarray([np.max(q) for q in np.abs(np.add(y_, -y))])
        elif type == 'variance':
            return np.std(np.abs(np.add(y_, -y)), axis=1)

        return
