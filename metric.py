from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from matplotlib.font_manager import FontProperties
from torch.nn.functional import cosine_similarity
import copy

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    # RGB
    color0 = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
    color1 = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)
    color2 = (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0)
    color3 = (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0)
    color4 = (1.0, 0.4980392156862745, 0.0, 1.0)
    color5 = (1.0, 1.0, 0.2, 1.0)
    color6 = (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)
    color7 = (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0)
    color8 = (0.6, 0.6, 0.6, 1.0)
    color9 = (0.3, 0.5, 0.4, 0.7)
    colorcenters = (0.1, 0.1, 0.1, 1.0)
    c = [color0, color1, color2, color3, color4, color5, color6, color7, color8, color9]
    for i in range(label.shape[0]):
        if label[i] >= 10:
            color = c[label[i]-10]
            plt.text(data[i, 0], data[i, 1], 'O', color=color,  # plt.cm.Set123
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            # print(label[i])
            color = c[label[i]]
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=color,  # plt.cm.Set123
                     fontdict={'weight': 'bold', 'size': 9})
    # plt.legend()
    plt.xlim(-0.005, 1.02)
    plt.ylim(-0.005, 1.025)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontdict={'family' : 'Times New Roman'}, fontproperties = FontProperties())
    return fig


def TSNE_PLOT(Z, Y, name=""):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    F = tsne.fit_transform(Z)  # TSNE features——>2D
    fig1 = plot_embedding(F, Y, name)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    now_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    plt.savefig('./save/{}.png'.format(now_time), dpi=500)
    plt.close(fig1)


def make_ps(x, centroids):
    s = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - centroids, 2), 2))
    s = (s.t() / torch.sum(s, 1)).t()
    p = s ** 2 / s.sum(0)
    return (p.t() / p.sum(1)).t(), s




def valid(model, device, dataset, view, data_size, class_num, cp_type):
    test_loader = DataLoader(
            dataset,
            batch_size=data_size,
            shuffle=False,
        )
    model.eval()
    pred_vectors = []
    Hs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
    labels_vector = []
    for step, (xs, y) in enumerate(test_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            if cp_type == 3:
                hs, _, _, _ = model.forward_mmi(xs)
            else:
                hs, _, _, _ = model(xs)
        for v in range(view):
            hs[v] = hs[v].detach()
            Hs[v].extend(hs[v].cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
    Ha_all = []
    for v in range(view):
        Ha_all.append(torch.from_numpy(Hs[v]))
    Ha_all_con = torch.cat(Ha_all, dim=1)

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(Ha_all_con.detach().cpu().numpy())
    labels = kmeans.predict(Ha_all_con.detach().cpu().numpy())
    cluster_centers = kmeans.cluster_centers_
    # Accuarcy = cluster_acc(labels, labels_vector)
    nmi, ari, acc, pur = evaluate(labels, labels_vector)
    # TSNE_PLOT(Ha_all[0].detach().cpu().numpy(), labels_vector, 'view 1')
    # TSNE_PLOT(Ha_all[1].detach().cpu().numpy(), labels_vector, 'view 2')
    # TSNE_PLOT(Ha_all_con.detach().cpu().numpy(), labels_vector, '')
    print('acc of complete part', acc)
    print('nmi of complete part', nmi)
    print('ari of complete part', ari)
    print('pur of complete part', pur)


