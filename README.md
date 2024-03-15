# 159 Assignment 2 - Sphereface

## 1. Paper Reading (30%)

### Contributions of The Paper

Two of the big contributions include:

- **Angular Softmax (A-Softmax) loss**: The paper introduced a new loss function specifically tailored to the task of face recognition called the Angular Softmax (A-Softmax) loss. The innovation with this loss function was that it allowed the network to learn angularly distributed feature embeddings, making the representations more discrimative for face recognition tasks.
- **Hypersphere Manifold**: In relation to the above, the paper also introduced the notion of embedding the features into a hypersphere manifold, as opposed to a traditional Euclidean space, helping improve the separability and compactness of the embeddings.

The researchers note that for face recognition, the features learned by a softmax loss have an intrinsic angular distribution, thus Euclidean margin constraints were not the best choice, hence the motivation for the A-Softmax loss.

### 3 Properties of the proposed A-Softmax

#### Property 1: Adjustable Angular Margin Difficulty

The A-Softmax loss is designed to incorporate an angular margin into the learning process, which introduces a geometric constraint. The angular margin is controlled by a parameter $m$. With larger $m$, the angular margin becomes larger, the constrained region on the manifold becomes smaller, and the corresponding
learning task also becomes more difficult. Therefore, there exists a minimal m that constrains the maximal intra-class angular distance to be smaller than the minimal inter-class angular distance

### Property 2: Lower Bound of $m$ in Binary Classification $m_{min} ≥ 2 + \sqrt{3}$

In binary classification, $m$ needs to be set to a value greater than or equal to $2 + \sqrt{3}$ to ensure this.

### Property 3: Lower Bound of $m$ in Multi-class Classification $m_{min} ≥ 3$

In multi-class classification, $m$ needs to be set to a value greater than or equal to 3 to ensure this.

### Description/Calculation of Metric Used With LFW Dataset

For multi-class classification against the LFW dataset, the paper trained a 64-layer CNN using A-Softmax loss with an $m$ of 4.

Beside reporting accuracy, the paper also proposed and used the "Angular Fisher score" for evaluating the feature discriminativeness. It is defined as:

$$
\text{AFS} = \frac{S_w}{S_b}
$$

Where the within-class scatter value $S_w$ is defined as:

$$
S_w = \sum_i \sum_{x_j \in X_i} n_i (1 - \cos(x_j, m_i))
$$

And the between-class scatter value $S_b$ is defined as:

$$
S_b = \sum_i n_i (1 - \cos(m_i, m))
$$

## 2. Implementation Results (70%)

### Criteria

| | | |
|---|---|---|
| Data loading and augmentation | 10% | ✅ |
| Design of neural networks | 15% | ✅ |
| Loss function | 10% | ✅ |
| Training  | 15% | ✅ |
| Testing and results | 10% | ✅ |
| Code format | 10% | ✅ |


For augmentation, the following transformations were used:

```python
transforms.Compose(
            [
                transforms.Resize(96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
```

The nural follows the 4-layer CNN architecture given in the paper:

```python
class SphereCNN(nn.Module):
    def __init__(self, class_num: int, feature=False):
        super(SphereCNN, self).__init__()
        self.class_num = class_num
        self.feature = feature

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=3, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=2
        )

        self.fc5 = nn.Linear(512 * 5 * 5, 512)
        self.angular = AngularPenaltySMLoss(512, self.class_num)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(
            self.conv4(x)
        )  # batch_size (0) * out_channels (1) * height (2) * width (3)

        x = x.view(x.size(0), -1)  # batch_size (0) * (out_channels * height * width)
        x = self.fc5(x)

        if self.feature:
            return x
        else:
            x_angle = self.angular(x, y)
            return x, x_angle
```

The loss function used is the A-Softmax loss:

```python
class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-7, m=None):
        super(AngularPenaltySMLoss, self).__init__()
        self.m = 4.0 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        numerator = torch.cos(
            self.m
            * torch.acos(
                torch.clamp(
                    torch.diagonal(wf.transpose(0, 1)[labels]),
                    -1.0 + self.eps,
                    1 - self.eps,
                )
            )
        )

        excl = torch.cat(
            [
                torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(labels)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        L = numerator - torch.log(denominator)

        return -torch.mean(L)
```

The results of my training and testing can be found in the following table documenting a training run of 100 epochs, calculating the accuracy on the LFW test dataset, every 20 epochs:

| Epoch | Loss   | Test Acc. |
|-------|--------|-----------|
| 04    | 6.836  |           |
| 09    | 6.675  |           |
| 14    | 6.516  |           |
| 19    | 6.334  | 0.5840    |
| 24    | 6.154  |           |
| 29    | 5.973  |           |
| 34    | 5.798  |           |
| 39    | 5.632  | 0.6230    |
| 44    | 5.457  |           |
| 49    | 5.282  |           |
| 54    | 5.123  |           |
| 59    | 4.940  | 0.6520    |
| 64    | 4.766  |           |
| 69    | 4.612  |           |
| 74    | 4.449  |           |
| 79    | 4.278  | 0.6510    |
| 84    | 4.123  |           |
| 89    | 3.961  |           |
| 94    | 3.782  |           |
| 99    | 3.657  | 0.6490    |

For training, an $m$ of 4 was used.

As you can see, overfitting appears to occur during the latter epochs.
