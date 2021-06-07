#%%
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from src.data.data import mnist
from src.models.model import cnn_model

_, testloader = mnist()

model = cnn_model()
model.load()

embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in testloader:

        n_batch = imgs.shape[0]
        _ = model.forward(imgs)
        embeddings.append(model.x1.data.numpy().copy())
        all_labels.extend(labels.data.numpy().copy())

embeddings = np.concatenate(embeddings)
print(embeddings.shape)
X_embedded = TSNE(n_components=2).fit_transform(
    embeddings[
        :1000,
    ]
)

sns.color_palette()
sns.scatterplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1], hue=all_labels[:1000], palette="tab10"
)
plt.savefig("../../reports/figures/t-SNE.png", dpi=300)
