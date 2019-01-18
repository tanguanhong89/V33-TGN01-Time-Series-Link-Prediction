import numpy as np


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


def for_looper(loop_function, loops_start_end_step, tensorify_result=False, _argv=[], _first_loop=True):
    #  loops_start_end_int = [[loop1_start,loop1_end,loop1_step],[loop2_start,loop2_end,loop2_step]]
    #  tensorify result, returns result as a tensor, ONLY WORKS WITH NUMBERS!!
    # DO NOT INITIALIZE argv!!!!, for recursive purposes
    # EXAMPLE
    # ex_looper = lambda input: return 'i:'+input[0]+'  j:'+input[1]
    # for_looper(loop_function=ex_looper, loops_start_end_step=[[1,5,1.3],[10,100,14]])

    start, end, step = loops_start_end_step[0]
    interval_values = np.arange(start, end, step)

    loop_results = []
    if _argv == []:
        _argv = [0 for x in range(len(loops_start_end_step))]
    for i in interval_values:
        argv_new = np.copy(_argv).astype(float)
        argv_new[len(_argv) - len(loops_start_end_step)] = i
        if len(loops_start_end_step) == 1:
            loop_results.append([argv_new, loop_function(argv_new)])
        else:
            loop_results += for_looper(loop_function=loop_function, loops_start_end_step=loops_start_end_step[1:],
                                       _argv=argv_new, _first_loop=False)
    if _first_loop and tensorify_result:
        tensor_positions_flattened = [x[0] for x in loop_results]
        loop_results_flattened = [x[1] for x in loop_results]

        dim_intervals = [np.arange(x[0], x[1], x[2]) for x in loops_start_end_step]
        steps_count_for_each_dim = [len(x) for x in dim_intervals]
        basic_tensor = np.zeros(steps_count_for_each_dim)
        result_tensor = np.copy(basic_tensor)

        tensor_positions_flattened_mapped = []
        for ele in tensor_positions_flattened:
            tensor_positions_flattened_mapped.append(
                [list(np.where(dim_intervals[i] == e)[0])[0] for i, e in enumerate(ele)])

        for i, mapped_pos in enumerate(tensor_positions_flattened_mapped):
            result_tensor[tuple(mapped_pos)] = loop_results_flattened[i]

        tensor_pos_interval_map = list(
            map(lambda x, y, z: [x, y, z], tensor_positions_flattened_mapped, tensor_positions_flattened,
                loop_results_flattened))
        return result_tensor, tensor_pos_interval_map

    return loop_results


class Optimisers:
    @staticmethod
    def n_ary_search_optimizer(score_function, search_space_argvs, search_resolution=4, convergence_limit=0.01,
                               max_iteration=3,
                               opti_obj='min', search_beyond_original_space=False):
        # score_function MUST return a dict with key 'score'
        # search space is an array of max,mins to used with score_function, eg [[x1min,x1max],[x2min,x2max]]
        # type is 'AUC' if wants a cluster of best extremas or 'best' for a single best extrema
        # EXAMPLE
        # score_fn = lambda x,y : return {'score': x*y*y-x*x, 'x': x, 'y':y}
        # n_ary_search_optimizer(score_function=score_fn, search_space_argvs=[[-30,30],[-10,10]])

        if opti_obj == 'min':
            obj_fn, rv_obj_rn = np.min, np.max
        elif opti_obj == 'max':
            obj_fn, rv_obj_rn = np.max, np.min
        else:
            raise TypeError('Please use "min" or "max" for opti_obj.')

        start_end_step_array = []
        for space in search_space_argvs:
            start, end = space
            step = (end - start) / search_resolution
            start_end_step_array.append([start, end, step])

        results = for_looper(score_function, loops_start_end_step=start_end_step_array,
                             tensorify_result=False)
        scores = [x[1]['score'] for x in results]
        best_score = obj_fn(scores)
        worst_score = rv_obj_rn(scores)

        best_pos = Hidden.parse_numpy_where_results(np.where(scores == best_score))[0]
        best_intervals = results[best_pos][0]

        if np.abs(best_score - worst_score) <= convergence_limit or max_iteration == 1:
            if max_iteration == 1:
                print('Max iter reached.')
            else:
                print('Results converged.')
            output = results[best_pos][1]
            output['best interval'] = best_intervals
            return results[best_pos][1]

        new_steps = [x[2] / 2 for x in start_end_step_array]

        new_max_space = best_intervals + new_steps
        new_min_space = best_intervals - new_steps

        if not search_beyond_original_space:
            new_max_space = [new_max_space[i] if new_max_space[i] < x[1] else x[1] for i, x in
                             enumerate(start_end_step_array)]
            new_min_space = [new_min_space[i] if new_min_space[i] > x[0] else x[0] for i, x in
                             enumerate(start_end_step_array)]

        new_search_space = [[new_min_space[i], new_max_space[i]] for i, x in enumerate(new_max_space)]

        return Optimisers.n_ary_search_optimizer(score_function=score_function, search_space_argvs=new_search_space,
                                                 search_resolution=search_resolution,
                                                 max_iteration=max_iteration - 1,
                                                 convergence_limit=convergence_limit, opti_obj=opti_obj,
                                                 search_beyond_original_space=search_beyond_original_space)

