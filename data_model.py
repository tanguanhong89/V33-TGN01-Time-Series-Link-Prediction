import pyorient
import time
from collections import Counter
import numpy as np


# unique ids are image (hash same as image), commandline

class ParentOfData:
    def __init__(self, usr, pw, host, db):
        # db_name = 'Blue_BlackPCv3'
        # usr = "root"
        # pw = "p@ssw0rd"
        self.usr = usr
        self.pw = pw
        self.host = host
        self.db = db
        return

    def retrieve_data(self):
        print("Connecting to the server...")
        client = pyorient.OrientDB(self.host, 2424)
        session_id = client.connect(user=self.usr, password=self.pw)
        print("OK - sessionID: ", session_id, "\n")

        if client.db_exists(self.db, pyorient.STORAGE_TYPE_PLOCAL):
            client.db_open(self.db, user=self.usr, password=self.pw)
            print('Getting ParentOf edges...')
            parentOf = client.command('select * from ParentOf')
            self.parentOfEdges = self._c(parentOf)

            self.parentOfVertices = {}

            a = time.time()
            s = ''
            k = []
            for e in self.parentOfEdges:
                o, i = str(e['out']), str(e['in'])
                k.append(o)
                k.append(i)
            k = list(Counter(k))
            k.sort()
            s = ','.join(k)
            print('Getting ParentOf vertices...')
            v = self._c(client.command('select * from [' + s + ']'))

            for i in range(len(k)):
                self.parentOfVertices[k[i]] = v[i]

        print('Done!')

        client.db_close()
        client.close()
        return

    def export_data(self, path):
        import pickle
        with open(path, 'wb') as file_object:  # save
            pickle.dump(obj=self, file=file_object)
        return

    def import_data(self, path):
        import pickle
        with open(path, 'rb') as file_object:  # load
            self = pickle.load(file_object)
        return self

    def construct_sequential(self, etc=[]):
        self._create_map_by_image()
        self._create_letters_map()
        time_sorted_events = []
        time_traveller = []

        for k in self.parentOfVertices:
            ele = self.parentOfVertices[k]
            eventTime = ele['EventTime']
            imageMap = list(filter(lambda x: x[1] == ele['Image'], self.map))[0][0]
            rid = k
            time_sorted_events.append([imageMap, rid, eventTime])

        time_sorted_events = sorted(time_sorted_events, key=lambda x: x[2])

        # process sequential parents
        parents = np.zeros(len(time_sorted_events), dtype=np.int32)
        parents = list(parents)

        ref_id = [x[1] for x in time_sorted_events]

        def swap(a, b, arr):
            t = arr[a]
            arr[a] = arr[b]
            arr[b] = t
            return arr

        for i, l in enumerate(self.parentOfEdges):
            parent_pos = ref_id.index(str(l['out']))  # out is parent
            current_pos = ref_id.index(str(l['in']))  # in is current

            if parent_pos > current_pos:
                if not time_sorted_events[parent_pos][2] == time_sorted_events[current_pos][2]:
                    print('Time traveller found:', str(l['in']), ', spawned from ', str(l['out']))
                    time_traveller.append(l['in'])
                else:
                    swap(current_pos, parent_pos, time_sorted_events)
                    swap(current_pos, parent_pos, ref_id)
                    swap(current_pos, parent_pos, parents)
                    current_pos, parent_pos = parent_pos, current_pos

            # every value in parents must be lower than its current position
            if parents[current_pos] == 0:
                parents[current_pos] = parent_pos
            else:
                print('Multiple parents for ', l['in'])
            pint = round(len(self.parentOfEdges) / 50)
            if i % pint == 0:
                print('Constructing sequential @ ', str(i * 100 / len(self.parentOfEdges))[:4], '%')

        print('Bringing back all time travellers...')
        for id in time_traveller:
            pos = ref_id.index(str(id))
            # time_sorted_events.pop(pos)
            parents[pos] = 0
        seq_labels = [x[0] for x in time_sorted_events]
        seq_time = [x[2] for x in time_sorted_events]
        seq_time = [self._total_seconds(x - seq_time[0]) for x in seq_time]

        seq_parents = parents

        seq_rids = [x[1] for x in time_sorted_events]

        seq_etc = []
        for id in seq_rids:
            tprop = []
            for p in etc:
                prop_value = self.parentOfVertices[id][p]
                if isinstance(prop_value, str):
                    prop_value = self._vectorize_string(prop_value)
                tprop.append(prop_value)
            seq_etc.append(tprop)
        self.seq_data = [seq_labels, seq_time, seq_parents, seq_etc]
        return self.seq_data

    def _create_map_by_image(self):
        # should be both image and commandline
        imagelist = [self.parentOfVertices[k]['Image'] for k in self.parentOfVertices]
        imagelist = list(Counter(imagelist))

        for i, d in enumerate(imagelist):
            imagelist[i] = [i, d]
        self.map = imagelist
        return self.map

    @staticmethod
    def _c(d):
        return [x.oRecordData for x in d]

    @staticmethod
    def _total_seconds(dt):
        total_seconds = dt.days * 24 * 60 * 60 + dt.seconds
        return total_seconds

    def _vectorize_string(self, s):
        vectors = [self.symb_map[l] for l in s]
        return vectors

    def _create_letters_map(self):
        symbs = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@#$%^&*()-_=+[]\{}|;\':",./<>? '
        cnt = len(symbs)
        emb = list(np.linspace(0, 1, cnt))
        self.symb_map = {}
        for i, v in enumerate(emb):
            self.symb_map[symbs[i]] = v
        return self.symb_map
