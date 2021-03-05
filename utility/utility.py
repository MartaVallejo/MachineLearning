import numpy as np
from sklearn import metrics
import re
####
# MATRIX PRINTING
####
REX = re.compile('\\033\[\d+m')


# utility function to print a line of a certain length per cell
# lng will be the number of 'cell'
# L will be the width of a cell
def print_line_matrix(lng, L=8):
    print('-' * ((L+1) * (lng) + 1))


# format a string so it fits in a cell
# cutted to L characters (L-1 +'\' actually)
# Numbers have thousands separator if possible
# string are centered.
# It's possible to right aligne numbers but I don't like it
def format_string(ele, L=8):
    ele = str(ele)
    colors = REX.findall(ele)
    value = sorted(REX.split(ele))[-1]
    if value.replace('.', '').isdigit():
        if value.isdigit():
            f_value = int(value)
        else:
            f_value = float(value)
        tmp = '{:,}'.format(f_value).replace(',', ' ')
        if len(tmp) < L:
            value = tmp
    if len(value) > L:
        value = value[:L-1]+'\\'
    value = value[:L].center(L)
    return ''.join(colors[:-1])+value+''.join(colors[-1:])


# function to format the row of a matrix
# r are the different cell
# L is the width for a cell
def format_row(r, L=8):
    return '|' + '|'.join([format_string(i, L) for i in r]) + '|'


# print a 2d array based on a layout
# each cell will have L characters
# can have color code
def print_matrix(layout, L=8):
    print_line_matrix(len(layout[0]), L)
    for i in range(len(layout)):
        print(format_row(layout[i], L))
        len_l = len(layout[i])
        if i + 1 < len(layout):
            len_l = max(len(layout[i+1]), len_l)
        print_line_matrix(len_l, L)


# get multiple values out of a confusion matrix
# recall (tpr)
# precition (ppv)
def get_score_main(matrix):
    res = {}
    res['tpr'] = matrix[0][0] / (matrix[:, 0].sum())
    res['ppv'] = matrix[0][0] / matrix[0].sum()
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
def get_score_predicted(matrix):
    res = {}
    res['tpr'] = matrix[0][0] / matrix[:, 0].sum()
    res['fnr'] = 1 - res['tpr']
    res['fpr'] = matrix[0][1] / matrix[:, 1].sum()
    res['tnr'] = 1 - res['fpr']
    return res


# get multiple values out of a confusion matrix
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
def get_score_label(matrix):
    res = {}
    res['ppv'] = matrix[0][0] / matrix[0].sum()
    res['fdr'] = 1 - res['ppv']
    res['for'] = matrix[1][0] / matrix[1].sum()
    res['npv'] = 1 - res['for']
    return res


# get multiple values out of a confusion matrix
# accuracy (acc)
# prevalence (pre)
def get_score_total(matrix):
    res = {}
    res['acc'] = sum(matrix.diagonal()) / matrix.sum()
    res['pre'] = matrix[:, 0].sum() / matrix.sum()
    return res


# get multiple values out of scores of a classification
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
def get_score_ratio(score):
    res = {}
    res['lr+'] = score['tpr'] / score['fpr'] if score['fpr'] != 0 else float('inf')
    res['lr-'] = score['fnr'] / score['tnr'] if score['tnr'] != 0 else float('inf')
    return res


# get the f1 value  out of scores of a classification
def get_score_f1(score):
    res = {}
    denom = (score['ppv'] + score['tpr'])
    if denom == 0:
        res['f_1'] = 0
    else:
        res['f_1'] = 2.0 *  (score['ppv'] * score['tpr']) / denom
    return res


# get multiple values out of scores of a classification
# f1 score (f_1)
# diagnostic odds ratio (dor)
def get_score_about_score(score):
    res = get_score_f1(score)
    if score['lr-'] == 0:
        res['dor'] = float('inf')
    else:
        res['dor'] = score['lr+'] / score['lr-']
    return res


# get all values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
# accuracy (acc)
# prevalence (pre)
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
# f1 score (f_1)
# diagnostic odds ratio (dor)
# area under the roc curve (auc)
def get_all_score(predicted, label, matrix):
    res = get_score_predicted(matrix)
    res = {**res, **get_score_label(matrix)}
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_ratio(res)}
    res = {**res, **get_score_about_score(res)}
    res['auc'] = metrics.roc_auc_score(label, predicted)
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# precision (ppv)
# accuracy (acc)
# prevalence (pre)
# f1 score (f_1)
# area under the roc curve (auc)
def get_score_verbose_2(predicted, label, matrix):
    res = get_score_main(matrix)
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_f1(res)}
    res['auc'] = metrics.roc_auc_score(label, predicted)
    return res


####
# Utility
####
blue = ['ppv', 'tpr', 'auc', 'f_1', 'acc', 'tnr']
yellow = ['npv', 'lr+']


# add color to a layout for pretty printing
# color the true in green and false in red
# color the value in the array above (blue, yellow) in blue or yellow
def add_color_layout(layout):
    layout[1][1] = bc.LGREEN + str(layout[1][1]) + bc.NC
    layout[2][2] = bc.LGREEN + str(layout[2][2]) + bc.NC
    layout[1][2] = bc.LRED + str(layout[1][2]) + bc.NC
    layout[2][1] = bc.LRED + str(layout[2][1]) + bc.NC
    # this should be a function somewhere, to much copy paste
    for i in range(0, min(len(layout), 4)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LCYAN + layout[ii][j] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LYELLOW + layout[ii][j] + bc.NC
    for i in range(4, len(layout)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LCYAN + layout[i][jj] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LYELLOW + layout[i][jj] + bc.NC


# append a series of value to the end of the first lines of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each row
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g],[h,i,j],[k,l,m]]
# inv: [0,1]
#
# layout = [[a,b,c,'acc','auc'],
#           [e,f,g,score['acc'],score['auc']],
#           [h,i,j,score['pre']],
#           [k,l,m,'pre']]
def append_layout_col(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele) - len(inv), dtype=int))
    for i in range(len(ele)):
        layout[i*2+(inv[i] % 2)] += ele[i]
        layout[i*2+((1+inv[i]) % 2)] += [score[j] for j in ele[i]]


# append a series of value to the end of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each col
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g]]
# inv: [0,1]
#
# layout = [[a,b,c],
#           [e,f,g],
#           ['acc', score['acc'], score['pre'], 'pre'],
#           [score['auc'],'auc']]
def append_layout_row(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele[0]) - len(inv), dtype=int))
    to_print = [[k for i in range(len(ele[j]))
                 for k in
                 ([ele[j][i], score[ele[j][i]]]
                 if inv[i] == 0 else
                 [score[ele[j][i]], ele[j][i]])]
                for j in range(len(ele))]
    layout.extend(to_print)


# clean a 2d array to make it ready for formatting
# round float and convert all element to string
def clean_layout(layout):
    layout = [[str(round(i, 3)) if isinstance(i, float) else str(i) for i in j]
              for j in layout]
    return layout


# print a comparison of the result of a classification
# label vs predicted
# it will print a confusion matrix
# verbose: how much measure are to be displayed (0,1,2,3)
# color: put color in
# L: cell width
def compare_class(predicted, label, verbose=1, color=True, L=8):
    unique_l = np.unique(label)[::-1]
    matrix = metrics.confusion_matrix(
        label, predicted, labels=unique_l).transpose()
    layout = [['pr\lb', *unique_l],
              [unique_l[0], *matrix[0]],
              [unique_l[1], *matrix[1]]]
    if (verbose > 0):
        layout.append(['total', matrix[:, 0].sum(),
                       matrix[:, 1].sum(), matrix.sum()])
        layout[0].append('total')
        layout[1].append(matrix[0].sum())
        layout[2].append(matrix[1].sum())
        if (verbose == 1):
            score = get_score_total(matrix)
            append_layout_col([['acc']], score, layout)
        elif (verbose == 2):
            score = get_all_score(predicted, label, matrix)
            append_layout_col([['ppv', 'acc'], ['f_1']], score, layout)
            append_layout_row([['tpr', 'tnr']], score, layout, inv=[0,1])
        elif (verbose == 3):
            score = get_all_score(predicted, label, matrix)
            append_layout_col([['ppv', 'fdr', 'acc'],
                               ['for', 'npv', 'pre']],
                              score, layout, inv=[0, 1])
            append_layout_row([['tpr', 'fpr', 'f_1'],
                               ['fnr', 'tnr', 'auc'],
                               ['lr+', 'lr-', 'dor']],
                              score, layout, inv=[0, 1, 0])
    layout = clean_layout(layout)
    if color:
        add_color_layout(layout)
    print_matrix(layout, L)

    
class bc:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LGRAY = '\033[37m'
    DGRAY = '\033[90m'
    LRED = '\033[91m'
    LGREEN = '\033[92m'
    LYELLOW = '\033[93m'
    LBLUE = '\033[94m'
    LMAGENTA = '\033[95m'
    LCYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    RBOLD = '\033[21m'
    UNDERLINE = '\033[4m'
    RUNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    RBLINK = '\033[25m'
    NC = '\033[0m'
