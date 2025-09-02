import torch
from network import Network, Encoder_imvc
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data, SubsetDataset
from scipy.optimize import linear_sum_assignment
import os
from metric import evaluate
from metric import TSNE_PLOT
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import copy
import itertools

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='BDGP')
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--mse_epochs", default=500)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--miss_rate", default=0.7)
parser.add_argument("--λ", default=0.2)
parser.add_argument("--cp_type", default=1, type=int,help="1: CL,  2: DD,  3: MI")
parser.add_argument("--non_iid", type=bool, default=False)
parser.add_argument("--build_your_own_dimvc", type=bool, default=False)
args = parser.parse_args()

if args.cp_type == 3:
    args.high_feature_dim = 64
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

if args.dataset == "BDGP":
    args.con_epochs = 10
    args.mmi_epochs = 50
    args.kl_epochs = 2000
    args.kl_interval = 1000
    args.learning_rate = 0.0003
    args.inc_epochs = 1000
    seed = 15
if args.dataset == "BBCSport":
    args.con_epochs = 5
    args.mmi_epochs = 500
    args.kl_epochs = 2000
    args.kl_interval = 1000
    args.batch_size = 128
    args.learning_rate = 0.0003
    args.inc_epochs = 100
    seed = 25
if args.dataset == "Hdigit":
    args.con_epochs = 20
    args.mmi_epochs = 50
    args.kl_epochs = 2000
    args.kl_interval = 500
    args.learning_rate = 0.001
    args.inc_epochs = 200
    seed = 15
if args.dataset == "Cifar10":
    args.con_epochs = 15
    args.mmi_epochs = 200
    args.kl_epochs = 2000
    args.kl_interval = 500
    args.learning_rate = 0.001
    args.inc_epochs = 500
    seed = 15
if args.dataset == "Cifar100":
    args.con_epochs = 50
    args.mmi_epochs = 200
    args.kl_epochs = 2000
    args.kl_interval = 500
    args.learning_rate = 0.0003
    args.inc_epochs = 500
    seed = 15
if args.dataset == "MNIST_USPS":
    args.con_epochs = 15
    args.mmi_epochs = 200
    args.kl_epochs = 2000
    args.kl_interval = 500
    args.learning_rate = 0.001
    args.inc_epochs = 500
    seed = 15
if args.dataset == "REU":
    args.con_epochs = 15
    args.mmi_epochs = 10
    args.kl_epochs = 2000
    args.kl_interval = 1000
    args.learning_rate = 0.0003
    args.inc_epochs = 1000
    seed = 15
if args.dataset == "NUSWIDE":
    args.con_epochs = 20
    args.mmi_epochs = 50
    args.kl_epochs = 1000
    args.kl_interval = 500
    args.learning_rate = 0.0003
    args.inc_epochs = 200
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


dataset, dims, view, data_size, class_num = load_data(args.dataset, args.miss_rate, seed, args.non_iid)
full_dataset = SubsetDataset(copy.deepcopy(dataset.X), copy.deepcopy(dataset.Y), view, args.miss_rate, data_size, isFull=True)
part_dataset = SubsetDataset(copy.deepcopy(dataset.X_i), copy.deepcopy(dataset.Y_i), view, args.miss_rate, data_size, isFull=False)

full_data_loader = torch.utils.data.DataLoader(copy.deepcopy(full_dataset), batch_size=args.batch_size, shuffle=True, drop_last=True if args.cp_type == 1 else False)

full_data_loader_maxbatch = torch.utils.data.DataLoader(copy.deepcopy(full_dataset), batch_size=len(full_dataset), shuffle=False, drop_last=False)

part_data_loader_maxbatch = torch.utils.data.DataLoader(copy.deepcopy(part_dataset), batch_size=len(part_dataset), shuffle=False, drop_last=False)


def pretrain(loader):
    model.train()
    for epo in range(args.mse_epochs):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            if args.cp_type == 3:
                _, _, xrs, _ = model.forward_mmi(xs)
            else:
                _, _, xrs, _ = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epo), 'Loss:{:.6f}'.format(tot_loss / len(loader)))


def contrastive_train(loader):
    model.train()
    for epo in range(args.con_epochs):
        tot_loss = 0.
        mse = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, qs, xrs, zs = model(xs)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(mse(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epo), 'Loss:{:.6f}'.format(tot_loss / len(loader)))

def make_ps(x, centroids):
    s = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - centroids, 2), 2))
    s = (s.t() / torch.sum(s, 1)).t()
    p = s ** 2 / s.sum(0)
    return (p.t() / p.sum(1)).t(), s

def kldiv_train(loader):
    model.train()
    for epo in range(args.kl_epochs):
        if epo % args.kl_interval == 0 :
            xs, _ = next(iter(loader))
            for v in range(view):
                xs[v] = xs[v].to(device)
            _, _, _, zs = model(xs)
            global_zs = torch.cat(zs, dim=1)
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            kmeans.fit(global_zs.detach().cpu().numpy())
            kl_centers = kmeans.cluster_centers_
            p_f, _ = make_ps(global_zs, torch.tensor(kl_centers).cuda())
        else:
            tot_loss = 0.
            mse = torch.nn.MSELoss()
            xs, _  = next(iter(loader))
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            _, qs, xrs, _ = model(xs)
            loss_list = []
            for v in range(view):
                mseloss = mse(xs[v], xrs[v])
                kl_loss = F.kl_div(F.log_softmax(qs[v], dim=1), F.softmax(p_f.clone().detach(), dim=1), reduction='batchmean')
                loss_list.append(kl_loss)
                loss_list.append(mseloss)
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            print('Epoch {}'.format(epo), 'Loss:{:.6f}'.format(tot_loss / len(loader)))


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def MMI(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss

def mmi_train(loader):
    model.train()
    for epo in range(args.mmi_epochs):
        tot_loss = 0.
        mse = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, _, xrs, _ = model.forward_mmi(xs)
            loss_list = []
            for vi in range(view):
                for ew in range(view):
                    if vi!=ew:
                        weight = 1 if vi == 0 and ew == 1 else 0.1
                        loss_list.append(weight * MMI(hs[vi], hs[ew]))
                loss_list.append(mse(xs[vi], xrs[vi]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epo), 'Loss:{:.6f}'.format(tot_loss / len(loader)))

def train_model_trans(t_model, batch_src, batch_tgt, optimizer):
    criterion_imvc = torch.nn.MSELoss()
    for _ in range(args.inc_epochs):
        optimizer.zero_grad()
        pred_tgt = t_model(batch_src)
        mse_loss = criterion_imvc(pred_tgt, batch_tgt)
        loss = mse_loss
        loss.backward()
        optimizer.step()

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, 100), nn.ReLU(), nn.Linear(100, num_classes))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()

        self.shared_layers = nn.Sequential(nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),

            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        return self.shared_layers(x)


class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1),  # 输出层为 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def compute_new_hs(train_hs_v, labels):
    unique_labels = torch.unique(labels)
    class_means = {}
    for label in unique_labels:
        indices = (labels == label)
        class_features = train_hs_v[indices]
        class_mean = class_features.mean(dim=0)
        class_means[label.item()] = class_mean
    return class_means


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_
        return grad_input, None


def train_cp(cp_type):
    if cp_type == 1:
        contrastive_train(full_data_loader)
    elif cp_type == 2:
        kldiv_train(full_data_loader_maxbatch)   # the data_loader MUST BE full_data_loader_maxbatch
    elif cp_type == 3:
        mmi_train(full_data_loader)
    else:
        raise ValueError("input error")

def prepare_settings():
    for v in range(view):
        feature_extractor = FeatureExtractor(args.high_feature_dim, args.high_feature_dim).to(device)
        classifier = Classifier(args.high_feature_dim, class_num).to(device)
        discriminator = Discriminator(args.high_feature_dim).to(device)

        feature_extractors.append(feature_extractor)
        classifiers.append(classifier)
        discriminators.append(discriminator)

        optimizers_feature_extractor.append(optim.Adam(feature_extractor.parameters(), lr=0.00005))
        optimizers_classifier.append(optim.Adam(classifier.parameters(), lr=0.0001))
        optimizers_discriminator.append(optim.Adam(discriminator.parameters(), lr=0.0001))

def get_com_labels():
    # Only one batch, can NOT shuffle
    for (xs, y), (xs_i, y_i) in zip(full_data_loader_maxbatch, part_data_loader_maxbatch):
        for v in range(view):
            xs[v] = xs[v].to(device)
            xs_i[v] = xs_i[v].to(device)
            y = y.to(device)
            y_i[v] = y_i[v].to(device)
        with torch.no_grad():
            if args.cp_type == 3:
                hs, _, _, _ = model.forward_mmi(xs)
                hs_i, _, _, _ = model.forward_mmi(xs_i)
            else:
                hs, _, _, _ = model(xs)
                hs_i, _, _, _ = model(xs_i)
        Ha_all = []
        for v in range(view):
            Ha_all.append(hs[v])
        Ha_all_con = torch.cat(Ha_all, dim=1)
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        kmeans.fit(Ha_all_con.detach().cpu().numpy())
        labels_prd = torch.as_tensor(kmeans.predict(Ha_all_con.detach().cpu().numpy())).long().to(device)
        return labels_prd

def transfer_training():
    print('Begin transfer training ... ')
    for epoch in range(args.inc_epochs):
        total_loss_views = 0
        for (xs, y), (xs_i, y_i) in zip(full_data_loader_maxbatch, part_data_loader_maxbatch):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs_i[v] = xs_i[v].to(device)
                y = y.to(device)
                y_i[v] = y_i[v].to(device)

            for v in range(view):
                if args.cp_type == 3:
                    hs, _, _, _ = model.forward_mmi(xs)
                    hs_i, _, _, _ = model.forward_mmi(xs_i)
                else:
                    hs, _, _, _ = model(xs)
                    hs_i, _, _, _ = model(xs_i)

                # Domain label: source domain  0, target domain  1
                lambda_ = 1  # The inverted gradient coefficient of the discriminator
                domain_labels_s = torch.zeros(len(hs[0]), 1).to(device)
                domain_labels_t = torch.ones(len(hs_i[0]), 1).to(device)

                # Feature extraction and classification
                features_s = feature_extractors[v](hs[v])
                predictions_s = classifiers[v](features_s)
                loss_classification = criterion_classification(predictions_s, labels_prd)

                # Discriminator training (source and target domains)
                features_t = feature_extractors[v](hs_i[v])
                domain_s = discriminators[v](GradientReversalLayer.apply(features_s, lambda_))
                domain_t = discriminators[v](GradientReversalLayer.apply(features_t, lambda_))

                loss_domain = criterion_discriminator(domain_s, domain_labels_s) + criterion_discriminator(domain_t, domain_labels_t)
                total_loss = loss_classification + loss_domain

                optimizers_feature_extractor[v].zero_grad()
                optimizers_classifier[v].zero_grad()
                optimizers_discriminator[v].zero_grad()
                total_loss.backward()
                optimizers_feature_extractor[v].step()
                optimizers_classifier[v].step()
                optimizers_discriminator[v].step()

                total_loss_views += total_loss

    # eval
    with torch.no_grad():
        pred_cp = labels_prd.detach().cpu().numpy()
        true_cp = y.detach().cpu().numpy()
        pred_icp = []
        true_icp = []
        for vi in range(view):
            pred_icp.append(torch.max(classifiers[vi](feature_extractors[vi](hs_i[vi])), 1)[1].detach().cpu().numpy())
            true_icp.append(y_i[vi].detach().cpu().numpy())
        all_preds = np.concatenate([pred_cp] + pred_icp)
        all_trues = np.concatenate([true_cp] + true_icp)
        overall_acc = cluster_acc(all_preds, all_trues)
        print('Overall accuracy by classification:', overall_acc)

    return  hs, hs_i, y, y_i


def impu_training():
    models_trans = []
    optimizers_trans = []
    avg_class = []
    for v in range(view):
        for u in range(view):
            # Create a transition model, such as view 1 to view 2, view 2 to view 1
            t_model = Encoder_imvc(args.high_feature_dim, args.high_feature_dim, class_num).to(device)
            models_trans.append((v, u, t_model))  # model v -> u
            t_optimizer = torch.optim.Adam(t_model.parameters(), lr=0.0001)
            optimizers_trans.append((v, u, t_optimizer))
    with torch.no_grad():
        train_hs = [feature_extractors[v](hs[v]).to(device) for v in range(view)]
        for v in range(view):
            avg_class.append(compute_new_hs(train_hs[v], labels_prd))

    # Train the converter model
    for (v, u, t_model), (_, _, t_optimizer) in zip(models_trans, optimizers_trans):
        # Train model v -> u
        if v != u:
            train_model_trans(t_model, train_hs[v], train_hs[u], t_optimizer)
    return models_trans, avg_class

def eval():
    # eval
    complete_H = []
    for u in range(view):
        view_features = []  # Used to store all features of view u
        # 1. Add view u's own features (complete)
        view_features.append(feature_extractors[u](hs[u]))  # hs for view u

        # 2. Traverse all view angles v and process the features corresponding to hs_i[v] in turn
        for v in range(view):
            if v == u:
                # If it is the current view u, append its own hs_i feature
                view_features.append(feature_extractors[u](hs_i[u]))  # hs_i for view u
            else:
                # Find the mapping model from view v to view u
                for (v_src, u_target, t_model) in models_trans:
                    if v_src == v and u_target == u:
                        # Map the features of view v to view u
                        transformed_features = t_model(feature_extractors[v](hs_i[v]))
                        labels_i_prd = classifiers[v](feature_extractors[v](hs_i[v])).argmax(dim=1).to(device)
                        avg_features = torch.stack([avg_class[u][lbl.item()] for lbl in labels_i_prd])
                        threshold = args.λ
                        view_feature = threshold * avg_features + (1 - threshold) * transformed_features
                        view_features.append(view_feature)  # 添加到列表中
        # 3. Merge all features
        complete_H.append(torch.cat(view_features, dim=0))
    complete_Y = torch.cat([y] + [y_i[i] for i in range(view)], dim=0)
    Ha_all = []
    for v in range(view):
        Ha_all.append(complete_H[v])
    Ha_all_con = torch.cat(Ha_all, dim=1)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(Ha_all_con.detach().cpu().numpy())
    labels = kmeans.predict(Ha_all_con.detach().cpu().numpy())
    cluster_centers = kmeans.cluster_centers_
    labels_vector = complete_Y.detach().cpu().numpy().reshape(len(complete_Y))
    nmi, ari, acc, pur = evaluate(labels, labels_vector)
    print('Final acc', round(100 * acc, 1))
    print('Final nmi', round(100 * nmi, 1))
    print('Final ari', round(100 * ari, 1))
    print('Final pur', round(100 * pur, 1))

if __name__ == '__main__':

    T = 1

    accs = []
    nmis = []
    purs = []

    for i in range(T):

        print("ROUND:{}".format(i + 1))

        if not args.build_your_own_dimvc:
            # initiate parameters
            setup_seed(seed)
            model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device).to(device)
            params = itertools.chain.from_iterable(model.ae_list[i].parameters() for i in range(view)) if args.cp_type == 3 else model.parameters()
            optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

            # Step1. Pre-train the complete data
            pretrain(full_data_loader)

            # *optional. Check the overall ACC of the training results of complete data
            valid(model, device, full_dataset, view, len(full_dataset), class_num, args.cp_type)

            # Step2.
            # Train the complete multi-view data
            # We provide three training methods: cl, dd, mi.
            # You can also add your own training methods.
            train_cp(args.cp_type)

        else:  # args.build_your_own_dimvc == True
            model = torch.load('./models/' + args.dataset + '.pth', map_location='cpu').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

        # *optional. Check the overall ACC of the training results of complete data
        valid(model, device, full_dataset, view, len(full_dataset), class_num, args.cp_type)


        #  Step3. Initialize perspective-related network components
        feature_extractors = []
        classifiers = []
        discriminators = []
        optimizers_feature_extractor = []
        optimizers_classifier = []
        optimizers_discriminator = []
        criterion_classification = nn.CrossEntropyLoss()  # classification loss
        criterion_discriminator = nn.BCELoss()  # discrimination loss
        prepare_settings()

        # Step4. Get the labels of the complete data after training.
        labels_prd = get_com_labels()

        # Step5. Transfer training between complete and incomplete data in each view
        hs, hs_i, y, y_i = transfer_training()

        # Step6. Imputation stage, training some functions to predict missing data
        models_trans, avg_class = impu_training()

        # Step7. Evaluate the final ACC, NMI, ARI and PUR.
        eval()


