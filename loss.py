import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

class contrastiveLoss(nn.Module):
    def __init__(self, base_temp = 0.07, temp = 0.07):
        super(contrastiveLoss, self).__init__()
        self.base_temp = base_temp
        self.temp      = temp
        
    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
       

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temp)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temp / self.base_temp) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
        




def calcualate_inner_outer_distances(features, labels):

    unique_labels = torch.unique(labels)

    class_centroids = []
    for label in unique_labels:
        class_mask = (labels == label)
        class_features = features[class_mask]
        class_centroid = torch.mean(class_features, dim=0)
        class_centroids.append(class_centroid)

    class_centroids = torch.stack(class_centroids) 


    inner_class_distances = torch.zeros(unique_labels.shape[0])
    for i, label in enumerate(unique_labels):
        class_mask = (labels == label)
        class_features = features[class_mask]
        class_centroid = class_centroids[i]
        distances = [euclidean(class_centroid.numpy(), feat.numpy()) for feat in class_features]
        inner_class_distances[i] = torch.tensor(distances).mean()

    outer_class_distances = []
    for i in range(unique_labels.shape[0]):
        for j in range(i + 1, unique_labels.shape[0]):
            distance = euclidean(class_centroids[i].numpy(), class_centroids[j].numpy())
            outer_class_distances.append(distance)

    outer_class_distances = torch.tensor(outer_class_distances)
    
    return inner_class_distances, outer_class_distances


def calculate_pca_analysis(features, labels): 
 
    pca2d = PCA(n_components = 2)
    principal_components2d = pca2d.fit_transform(features)
    
    
    return  principal_components2d

def plot_pca(principal_components2d,labels, idx2class, dir_path, idx):
    fig1 = plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    color_map = cm.get_cmap('viridis', len(unique_labels))
    colors = [color_map(label) for label in labels]
    label_color_dict = {idx2class[label]: color_map(i) for i, label in enumerate(unique_labels)}


    pc1 = principal_components2d[:, 0]
    pc2 = principal_components2d[:, 1]


    plt.scatter(pc1, pc2, c=colors)
    plt.title('2D PCA Visualization')

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label) for label, color in label_color_dict.items()]
    plt.legend(handles=handles, title='Labels')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    fig1.savefig(f'{dir_path}/{idx}_2d.png')
    plt.close(fig1)
    
def plot_pca_2(all_labels, pca_points, class_descriptions, path, epoch):
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    cmap = ListedColormap(custom_colors)


    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_points[:, 0], pca_points[:, 1], c=all_labels, cmap=cmap)


    legend_handles = [Line2D([0], [0], marker='o', color='w', label=class_descriptions[label], markersize=8, markerfacecolor=color, markeredgecolor='k') for label, color in zip(np.unique(all_labels), custom_colors)]

    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.title('PCA Projection with Custom Colored Labels and Class Descriptions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    fig.savefig(f'{path}/pca_{epoch}.png')
    plt.close(fig)

def testing_metrics(loader, batch_size, model, epoch,device, path = '/home/andrii/adient/debug_output/pca'):
    all_features = torch.zeros((1,model.n_embeddings))
    all_labels = torch.tensor([])

    for image, label in loader:
        images = torch.cat([image[0], image[1]], dim = 0).to(device)
        features = model(images)
        features2 = features.cpu()
        all_features = torch.cat([all_features, features2[:batch_size, :]], dim = 0)
        all_labels = torch.cat([all_labels, label])
    all_features = all_features[1:,:]   
    inner_d, outer_d = calcualate_inner_outer_distances(all_features.detach(), all_labels.detach())
    pca3, pca2 = calculate_pca_analysis(all_features.detach(), all_labels.detach())
    plot_pca(pca2, pca3, all_labels.detach(), loader.dataset.idx2class, path, epoch)
    
    return inner_d, outer_d

if __name__ == '__main__': 
    pass
    #Usage example
    # all_features = torch.zeros((1,384))
    # all_labels = torch.tensor([])
    # for images, labels in test_dataloader:
    #     images = torch.cat([images[0], images[1]], dim=0).to(device)
    #     features = backbone(images)
    #     features = features.cpu()
    #     all_features = torch.cat([all_features, features[:5, :]], dim = 0 )
    #     all_labels = torch.cat([all_labels, labels])

    # all_features = all_features[1:,:]

    # pca3d, pca2d = calculate_pca_analysis(all_features, all_labels)
    # plot_pca(pca2d, pca3d, all_labels, test_dataloader.dataset.idx2class)