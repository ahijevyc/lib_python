def limits(stat):
    x = {
        'BRIER'   : [ 0   ,0.15],
        'CSI'     : [ 0   ,1],
        'FSS'     : [ 0   ,1],
        'GSS'     : [ 0   ,0.8],
        'BAGSS'   : [ 0   ,0.8], # bias adjusted GSS
        'HK'      : [-0.2 ,1],
        'HSS'     : [-0.2 ,1],
        'BSS_SMPL': [-1   ,1],
        'ROC_AUC' : [ 0.45,1],
        }
    if stat not in x.keys():
        return [0,1]
    return x[stat]
