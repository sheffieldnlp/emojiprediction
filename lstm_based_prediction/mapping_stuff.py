# coding: utf-8

import numpy as np

def mapping(filename):
    x = np.load(filename)

    mapping = {'11': 18, '10': 13, '13': 7, '12': 8, '15': 16, '14': 11, '17': 1, '16': 12, '19': 19, '18': 3, '1': 4, '0': 2, '3': 9, '2': 0, '5': 10, '4': 14, '7': 17, '6': 15, '9': 5, '8': 6}

    rmap = dict((v,k) for k,v in mapping.iteritems())

    nx = []
    for i in x:
        nx.append(rmap[i])

    with open(filename+'_preds.txt', 'w') as f:
        for i in nx:
            f.write('%s\n' % i)

if __name__ == '__main__':
    import plac
    plac.call(mapping)
