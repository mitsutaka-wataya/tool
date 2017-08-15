import itertools as it
from . import base

def comb(size, choise_1st):
    fullset = set(range(size))
    l = []
    for i in it.combinations(fullset, choise_1st):
        l += [set(i)]
    return l

def multivariate_mi_ind(mi_dict, response):
    celllist = response.response
    fullset = response.fullset()
    N = len(fullset)
    tmpmilist = n1list(mi_dict, fullset)
    result_dict = tmpmilist[-1]
    for n in range(2, N+1):
        tmpmilist = mi_nlist(n, tmpmilist, fullset)
        result_dict.update(tmpmilist[-1])
    return result_dict

def show_multivariate_ind(multi_mi_ind):
    for i in multi_mi_ind:
        print('I(' + '; '.join(map(str, multi_mi_ind[i][0])) + '| ' + str(i).strip('(').strip(')') + ') = ' + str(multi_mi_ind[i][1]))

def n1list(mi_dict, fullset):
    def n1_mi_m(mi_dict, fullset, m):       
        rest = comb(len(fullset), m+1)
        if m == 0:
            return {'':((0,), mi_dict[(0,)])}
        numCombination = len(comb(len(fullset), m))
        n1_mi_dict = {}
        known_set = set()
        for i in range(numCombination):
            joint_mi_pair = {tuple(v-{i}):tuple(v) for v in rest if {i}.issubset(v) and (not {tuple((v-{i}))}.issubset(known_set))}
            n1_mi_dict.update({v:(tuple(set(joint_mi_pair[v])-set(v)), mi_dict[joint_mi_pair[v]] - mi_dict[v]) for v in joint_mi_pair})
            known_set = known_set | set(list(joint_mi_pair.keys()))
        return n1_mi_dict

    mlist = range(len(fullset))
    return [n1_mi_m(mi_dict, fullset, m) for m in mlist]

def mi_nlist(n, n_minus1_list, fullset):
    milist = []
    numParam = len(fullset)
    for m in range(numParam - n +1):
        n_minus1_m = n_minus1_list[m]
        n_minus1_mplus1 = n_minus1_list[m+1]
        tmpDict = {}
        mi_pair = {}
        known_set = set()
        if m == 0:
            milist += [{'':(tuple(set(n_minus1_m[''][0]) | {n-1}) , n_minus1_m[''][1] - n_minus1_mplus1[(n-1,)][1])}]
        else:
            for i in range(n-1, numParam):
                tmpDict.update({v[0]: v[1] for v in it.product(list(n_minus1_m.keys()), list(n_minus1_mplus1.keys())) if (set(v[0])|{i}) == set(v[1]) and not ({i}.issubset(set(n_minus1_m[v[0]][0]))) and not ({v[0]}.issubset(known_set))})
                known_set = set(tmpDict.keys()) | known_set
            milist += [{v: (tuple(set(tmpDict[v])-set(v) | set(n_minus1_m[v][0])), n_minus1_m[v][1]-n_minus1_mplus1[tmpDict[v]][1]) for v in tmpDict}]
    return milist