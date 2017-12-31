import numpy as np
import time
import math

from mnist2ndarray import *

IMGSIZE=28
CMAX=255

DO_PLOT = True 

# numpy
float_formatter = lambda x: "%9.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter},
    linewidth=120, threshold=np.inf)

##############################

##############################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

ax_es_opts = {
    "mean": {
        "gpos": 2,
        "gposi": np.s_[5:7],
        "size": (IMGSIZE,IMGSIZE),
        "cmap": "gray", 
        "vmin": 0.0, 
        "vmax": 1.0}, 
    "std": {
        "gpos": 3,
        "gposi": np.s_[7:9],
        "size": (IMGSIZE,IMGSIZE),
        "cmap": "afmhot", 
        "vmin": 0.0, 
        "vmax": 1.0}, 
    "cov": {
        "gpos": 4,
        "gposi": np.s_[9:11],
        "size": (IMGSIZE*IMGSIZE,IMGSIZE*IMGSIZE),
        "cmap": "br1",
        "vmin": -0.1, 
        "vmax": 0.1}} 

cmap_br1 = {'red':  ((0.0, 1.0, 1.0),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.2, 0.2)),
         'green':   ((0.0, 0.2, 0.2),
                    (0.5, 0.0, 0.0),
                    (1.0, 0.8, 0.8)),
         'blue':    ((0.0, 0.25, 0.25),
                    (0.5, 0.0, 0.0),
                    (1.0, 1.0, 1.0))}

br1 = LinearSegmentedColormap('br1', cmap_br1)
plt.register_cmap(cmap=br1)

PLT_PAUSETIME=0.1

#
def fig_init():

    # bar, eimage, label, evidence
    global fig, gs, ax_b, ax_i, ax_l, ax_es, ax_es_i, n_labels
    global f_im, f_la, f_es, f_es_i

    fig = plt.figure(figsize=(32, 18))
    fig.patch.set_facecolor('#000000')

    gs = gridspec.GridSpec(5, 1+n_labels, height_ratios=[0.3,3,1,1,1])
    gs.update(wspace=0.05, hspace=0.02, 
        left=0.1, right=0.9, bottom=0.1, top=0.9)

    for i, desc in enumerate(("trained", "training", 
        "mean", "standard\ndeviation", "covariance")):
        ax = plt.subplot(gs[i,0])
        ax.set_axis_off()
        ax.text(0.8, 0.5, desc,
            verticalalignment='center', horizontalalignment='right',
            transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,4])
    ax.set_axis_off()
    ax.text(0.5, 0.5, ">",
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, fontsize=30, color="#999999")

    # 프로그레스바
    ax_b = plt.subplot(gs[0,1:])
    ax_b.set_axis_off()


    # 학습중인 이미지
    ax_i = plt.subplot(gs[1,2:4])
    ax_i.set_axis_off()
    f_im = ax_i.imshow(np.zeros((IMGSIZE,IMGSIZE)),
        interpolation='none', cmap="gray", vmin=0, vmax=1.0)

    # 학습중인 라벨
    ax_l = plt.subplot(gs[1,1])
    ax_l.set_axis_off()
    f_la = None

    ax_es_i = {}
    f_es_i = {}
    for n in ax_es_opts:
        ax_es_i[n] = plt.subplot(gs[1,ax_es_opts[n]['gposi']])
        f_es_i[n] = ax_es_i[n].imshow(np.zeros(ax_es_opts[n]['size']),
            interpolation='none', cmap=ax_es_opts[n]['cmap'], 
            vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])
        ax_es_i[n].set_axis_off()

    ax_es = {}
    f_es = {}
    for i in range(n_labels):
        ax_es[i] = {}
        f_es[i] = {}
        for n in ax_es_opts:
            ax_es[i][n] = plt.subplot(gs[ax_es_opts[n]['gpos'],i+1])
            f_es[i][n] = ax_es[i][n].imshow(np.zeros(ax_es_opts[n]['size']),
                interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])

            ax_es[i][n].set_axis_off()

    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)


#
def fig_learn(trained, img, label, labels=[]):

    global f_im, f_la

    if (len(labels)==0):
        labels = [label]

    ax_b.barh(0, trained, facecolor="#33ccff",align="center")
    ax_b.set_xlim((0,m))
    ax_b.set_ylim((0,0))

    if (label != None):
        if (f_im != None): f_im.remove()
        img_2d = img.reshape(IMGSIZE,IMGSIZE)
        f_im = ax_i.imshow(img_2d, interpolation='none',
                cmap="gray", vmin=0, vmax=1.0)

        if (f_la != None): f_la.remove()
        f_la = ax_l.text(0.5, 0.5, "%s" % label,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_l.transAxes, fontsize=50, color="white")

    img_es = {}
    for n in ax_es_opts:

        if (label != None):
            img_es[n] = EoH[label][n].reshape(ax_es_opts[n]['size'])
            f_es_i[n].remove()
            f_es_i[n] = ax_es_i[n].imshow(img_es[n], 
                interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])

        for l in labels:
            if (l in EoH):
                img_es[n] = EoH[l][n].reshape(ax_es_opts[n]['size'])
                f_es[l][n].remove()
                f_es[l][n] = ax_es[l][n].imshow(img_es[n], 
                    interpolation='none', cmap=ax_es_opts[n]['cmap'], 
                    vmin=ax_es_opts[n]['vmin'], vmax=ax_es_opts[n]['vmax'])


    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)

#
def fig_init_test():

    global fig, gs, ax_b, ax_i, ax_l, ax_a
    global f_im, f_la, f_an

    ax = plt.subplot(gs[0:2,:])
    ax.cla()

    ax = plt.subplot(gs[0,0])
    ax.set_axis_off()
    ax.text(0.8, 0.5, "tested",
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,0])
    ax.set_axis_off()
    ax.text(0.8, 0.5, "testing",
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes, fontsize=30, color="#999999")

    ax = plt.subplot(gs[1,4])
    ax.set_axis_off()
    ax.text(0.5, 0.5, ">",
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes, fontsize=30, color="#999999")

    ax_b = plt.subplot(gs[0,1:])
    ax_b.set_axis_off()

    ax_i = plt.subplot(gs[1,2:4])
    ax_i.set_axis_off()
    f_im = ax_i.imshow(np.zeros((IMGSIZE,IMGSIZE)),
        interpolation='none', cmap="gray", vmin=0, vmax=1.0)

    ax_l = plt.subplot(gs[1,1])
    ax_l.set_axis_off()
    f_la = None

    ax_a = {}
    f_an = {}
    for i in (0,1,2):
        ax_a[i] = plt.subplot(gs[1,5+i*2:7+i*2])
        f_an[i] = None
        ax_a[i].set_axis_off()


    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)


#
COLOR_SUCCESS = "#66ff33"
COLOR_FAIL = "#ff3366"
def fig_test(success, fail, img=None, label=None, ds=[]):

    global fig, gs, ax_b, ax_i, ax_l, ax_a
    global f_im, f_la, f_an
    global m_test

    ax_b.barh(0,success,facecolor=COLOR_SUCCESS,align="center")
    ax_b.barh(0,fail,left=success,facecolor=COLOR_FAIL,align="center")
    ax_b.set_xlim((0,m_test))
    ax_b.set_ylim((0,0))

    if (label != None):
        if (f_im != None): f_im.remove()
        img_2d = img.reshape(IMGSIZE,IMGSIZE)
        f_im = ax_i.imshow(img_2d, interpolation='none',
                cmap="gray", vmin=0, vmax=1.0)

        if (f_la != None): f_la.remove()
        f_la = ax_l.text(0.5, 0.5, "%s" % label,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_l.transAxes, fontsize=50, color="white")

    for i, a in enumerate(ds[0:3]):
        if (i == 0):
            c = COLOR_SUCCESS if a == label else COLOR_FAIL
            fs = 150
        elif (a == label):
            c = "white"
            fs = 150
        else:
            c = "#999999"
            fs = 50

        if (f_an[i] != None): f_an[i].remove()
        f_an[i] = ax_a[i].text(0.5, 0.5, "%s" % a,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_a[i].transAxes, fontsize=fs, color=c)

        
    plt.show(block=False)
    plt.pause(PLT_PAUSETIME)

# Naive Bayes: P(H|E) = P(E|H)/P(E) * P(H)
# 
#   P(H|E):         posterior probability
#       
#
#   P(E|H):         likelihood
#       multivariate normal distribution
#       P(E|H) = N(E, EoH_mean,EoH_cov)
#
#   P(H):           prior probability
#       (count(H)+1)/(count(all H)+1)
#
#   P(E):           evidence
#
EoH = dict()
Using_Features = np.ones(IMGSIZE*IMGSIZE, dtype=bool)

def update_EoH(labels=[]):
    t = 0

    if (len(labels)==0):
        labels = LABELS

    for d in labels:
        d_imgs = train_images[np.all([train_labels==d, trained], axis=0)]

        if (len(d_imgs)==0):
            continue

        if (len(d_imgs)>1):
            cov = np.cov(d_imgs[...,Using_Features].T)
        else:
            cov = np.zeros((IMGSIZE**2,IMGSIZE**2))

        EoH[d] = {
            "mean" : np.mean(d_imgs[...,Using_Features],axis=0),
            "std" : np.std(d_imgs[...,Using_Features],axis=0),
            "cov" : cov
        }


##############################
# 
##############################
#
train_images_2d = mnist2ndarray("data/train-images-idx3-ubyte")/CMAX
train_images = train_images_2d.reshape(len(train_images_2d),-1)
train_labels = mnist2ndarray("data/train-labels-idx1-ubyte")
trained = np.zeros(len(train_images), dtype=bool)

LABELS = np.unique(train_labels)
n_labels = len(LABELS)
assert len(train_images) == len(train_labels)

m = len(train_images)
m = int(m/1) 

if DO_PLOT: fig_init()

FIG_LEARN_BOOTSTRAP=100
FIG_LEARN_INTERVAL=100
for i in range(m):
    trained[i] = True

    if (DO_PLOT and
        ((i < FIG_LEARN_BOOTSTRAP) or (i%FIG_LEARN_INTERVAL==0))):
            update_EoH()
            fig_learn(i, train_images[i], train_labels[i], LABELS)

l, C_H = np.unique(train_labels, return_counts=True)
P_H = (C_H+1) / (sum(C_H)+1)

update_EoH()
if DO_PLOT: fig_learn(m, None, None, LABELS)


for d in LABELS:
    Min_CovRow = np.median(abs(EoH[d]["cov"][abs(EoH[d]["cov"])>0.0]))
    for i, row in enumerate(EoH[d]["cov"]):
        if (np.median(abs(row))<Min_CovRow/50):
            Using_Features[i] = False

print ("Using_Features:",sum(Using_Features))
update_EoH()


#F_STD = np.std(train_images,axis=0)
#for i, var in enumerate(F_STD):
#    if (var < 0.2):
#        Using_Features[i] = False
#
#print ("Using_Features:",sum(Using_Features))
#EoH = make_EoH()

##############################

##############################
test_images_2d = mnist2ndarray("data/t10k-images-idx3-ubyte")/CMAX
test_images = test_images_2d.reshape(len(test_images_2d),-1)
test_labels = mnist2ndarray("data/t10k-labels-idx1-ubyte")
assert len(test_images) == len(test_labels)


m_test = len(test_images)
m_test = int(m_test/100) # 디버깅시에 일부만 가지고 하면 편함


test_results = np.zeros(m_test, dtype=bool)

test_results_2nd = np.zeros(m_test, dtype=bool)

# multivariate normal distribution
# P(E|H) = N(E, EoH_mean,EoH_cov)
def logN(img,mean,cov):

    const = 0.0

    detsign,logdet = np.linalg.slogdet(cov)
#    assert logdet > 0

    ln = -1/2*(logdet+
        np.dot((img-mean).T,np.linalg.inv(cov)).dot(img-mean)+const)

    return ln 


if DO_PLOT: fig_init_test()
    
FIG_TEST_BOOTSTRAP=100
FIG_TEST_INTERVAL=10

for i in range(m_test):

    P_HoE = np.zeros(n_labels)
    img_org = test_images[i]
    img = test_images[i,Using_Features]
    l = test_labels[i]
    for j in LABELS:
        P_HoE[j] = (np.log(P_H[j]) + 
                (logN(img, EoH[j]["mean"], EoH[j]["cov"])))
            
    ds = np.argsort(P_HoE)[-5:][::-1]
    d = ds[0] # 결과

    if (d==l):
        test_results[i] = True
    
    successes = sum(1*test_results)
    tests = (i+1)

    print ("%d / %d (%3.1f)" % (successes, tests, successes/tests*100.0))
    
    if (DO_PLOT and 
        ((i < FIG_TEST_BOOTSTRAP) or 
        (i%FIG_TEST_INTERVAL==0) or i==m_test-1)): #마지막
        fig_test(successes, tests-successes, img_org, l, ds)

time.sleep(5)