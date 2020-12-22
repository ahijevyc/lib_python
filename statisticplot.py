import pdb
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn import metrics

def limits(stat):
    x = {
        'BRIER'   : [ 0   ,0.15],
        'CSI'     : [ 0   ,1],
        'FSS'     : [ 0   ,1],
        'GSS'     : [ 0   ,1],
        'BAGSS'   : [ 0   ,1], # bias adjusted GSS
        'HK'      : [-0.2 ,1],
        'HSS'     : [-0.2 ,1],
        'BSS_SMPL': [-1   ,1],
        'ROC_AUC' : [ 0.45,1],
        }
    if stat not in x.keys():
        return [0,1]
    return x[stat]

def bss(obs, fcst):
    bs = np.mean((fcst-obs)**2)
    climo = np.mean((obs - np.mean(obs))**2)
    return 1.0 - (bs/climo)

def reliability_diagram(ax, obs, fcst, base_rate=None, label="", n_bins=10, debug=False, **kwargs):
    # allow lists
    obs = np.array(obs)
    fcst = np.array(fcst)
    # calibration curve
    true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=n_bins)
    one2oneline_label = "Perfectly calibrated"
    # If it is not a child already add perfectly calibrated line
    has_one2oneline = one2oneline_label in [x.get_label() for x in ax.get_lines()]
    if not has_one2oneline:
        one2oneline = ax.plot([0, 1], [0, 1], "k:", alpha=0.7, label=one2oneline_label)
    bss_val = bss(obs, fcst)
    if base_rate is None:
        base_rate = obs.mean() # base rate
    p_list = ax.plot( fcst_prob, true_prob, "s-", label="%s (%1.4f)" % (label, bss_val), **kwargs)
    p = p_list[0]
    noresline          = ax.axhline(y = base_rate, color=p.get_color(), linewidth=0.5, linestyle="dashed", dashes=(9,9))
    noresline_vertical = ax.axvline(x = base_rate, color=p.get_color(), linewidth=0.5, linestyle="dashed", dashes=(9,9))
    noskill_line = ax.plot([0, 1], [base_rate/2, (1+base_rate)/2], color=p.get_color(), linewidth=0.5)
    for x, f in zip(fcst_prob, true_prob):
        if np.isnan(f): continue # avoid TypeError: ufunc 'isnan' not supported...
        # label raw counts
        ax.annotate("%1.4f" % f, xy=(x,f), xycoords=('data', 'data'), 
            xytext = (0,1), textcoords='offset points', va='bottom', ha='center',
            fontsize='xx-small')
    return p_list
   
def count_histogram(ax, obs, fcst, label="", n_bins=10, debug=False):
    # Histogram of counts
    h = ax.hist(fcst, bins=n_bins, label=label, histtype='step', lw=2, alpha=1, log=True)
    return h

def ROC_curve(ax, obs, fcst, label="", n_bins=10, debug=False):
    # ROC auc
    if debug:
        print("auc", auc)
    ax.set_title("ROC curve")
    no_skill_label = "no skill"
    # If it is not a child already add perfectly calibrated line
    has_no_skill_line = no_skill_label in [x.get_label() for x in ax.get_lines()]
    if not has_no_skill_line:
        no_skill_line = ax.plot([0, 1], [0, 1], "k:", alpha=0.7, linewidth=0.8, label=no_skill_label)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xlabel("POFD")
    ax.set_ylabel("PODY")
    true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=n_bins)
    auc = metrics.roc_auc_score(obs, fcst)
    pofd, pody, _ = metrics.roc_curve(obs, fcst)
    r = ax.plot(pofd, pody, marker="+", linestyle="solid", label="%s (%1.4f)" % (label, auc))
    auc = ax.fill_between(pofd, pody, alpha=0.2)
    for s, x, y in zip(fcst_prob, pofd, pody):
        # label thresholds on ROC curve
        ax.annotate("%1.4f" % s, xy=(x,y), xycoords=('data', 'data'),
                xytext=(0,1), textcoords='offset points', va='baseline', ha='left',
                fontsize = 'xx-small')
    return r, auc
