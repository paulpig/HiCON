import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from re import compile,findall,split
import pdb
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
import random
import matplotlib as mpl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from matplotlib import gridspec
#import pickle5 as pickle
from sklearn.preprocessing import normalize
import seaborn as sns

print(mpl.__version__)

model_name = 'CrossRec'
# model_name = 'KGAT'
# print(model_name)

# user_emb = np.load('./{}/user_emb_hop_7.npy'.format(model_name))
# item_emb = np.load('./{}/item_emb_hop_7.npy'.format(model_name))

user_emb = np.load('./{}/user_emb.npy'.format(model_name))
item_emb = np.load('./{}/item_emb.npy'.format(model_name))


# normalization
user_emb = preprocessing.normalize(user_emb, norm='l2')
item_emb = preprocessing.normalize(item_emb, norm='l2')



fig = plt.figure(figsize=(9, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

n_components = 2
perplexity = 60

tsne = manifold.TSNE(
    n_components=n_components,
    init="random",
    random_state=1,
    perplexity=perplexity,
    learning_rate="auto",
    n_iter=600,
)

item_emb_2d = tsne.fit_transform(item_emb)
item_emb_2d = normalize(item_emb_2d, axis=1,norm='l2')

# pdb.set_trace()

cmap = plt.cm.get_cmap('BuPu', 16)
cmaplist = [cmap(i) for i in range(cmap.N)]

# cmaplist = cmaplist[-15:]
cmaplist = cmaplist[:7] + cmaplist[-8:]
cmaplist[0] = (1., 1., 1., 1.0)
# create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, len(cmaplist))
# define the bins and normalize
#bounds = np.linspace(0, 20, 21)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#sns.set_style("darkgrid")
#sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
kwargs = {'levels': np.arange(0, 4.2, 0.5)}
axs = plt.subplot(gs[0])
sns.kdeplot(data=item_emb_2d, bw=0.05, shade=True, cmap=cmap, ax=axs, legend=True, **kwargs)

#sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap=cmap, norm=norm, legend=True, **kwargs)
#sns.kdeplot(data=user_emb_2d, bw=0.05, shade=True, cmap="GnBu", legend=True, **kwargs)
#sns.kdeplot(data=user_emb_ori, bw=0.05, shade=True, cmap="GnBu", ax=axs[0][0], legend=True, **kwargs)
#sns.kdeplot(data=user_emb_hop, bw=0.05, shade=True, cmap="GnBu", ax=axs[0][1], legend=True, **kwargs)

my_x_ticks = np.arange(-1, 1.2, 1)
axs.set_xticks(my_x_ticks)
# axs.set_xticklabels(my_x_ticks, fontsize=30)
my_y_ticks = np.arange(-1, 1.2, 1)
axs.set_yticks(my_y_ticks)
# axs.set_yticklabels(my_y_ticks, fontsize=30)
axs.tick_params(axis='both', labelsize=23)

axs.spines['bottom'].set_linewidth(2.2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2.2);####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2.2);###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2.2);####设置上部坐标轴的粗细
axs.set_xlabel("Features", fontsize=30, fontdict={'family': 'Times New Roman'})
axs.set_ylabel("Features", fontsize=30, fontdict={'family': 'Times New Roman'}, labelpad=-5.5)

#KDE
x = [p[0] for p in item_emb_2d]
y = [p[1] for p in item_emb_2d]
angles = np.arctan2(y,x)
axs = plt.subplot(gs[1])
# sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True,ax=axs[1][4],color='green')
sns.kdeplot(data=angles, bw=0.15, shade=True,legend=True, ax=axs, color='blue')

# my_x_ticks = np.arange(-1, 1.2, 1)
# axs.set_xticks(my_x_ticks)

my_y_ticks = np.arange(0, 0.6, 0.2)
axs.set_yticks(my_y_ticks)
axs.tick_params(axis='both', labelsize=23)

axs.spines['bottom'].set_linewidth(2.2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2.2);####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2.2);###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2.2);####设置上部坐标轴的粗细
axs.set_xlabel("Angles", fontsize=30, fontdict={'family': 'Times New Roman'})
axs.set_ylabel("Density", fontsize=30, fontdict={'family': 'Times New Roman'}, labelpad=-1.5)

plt.tight_layout()
plt.savefig('./{}/visual.pdf'.format(model_name))
plt.show()