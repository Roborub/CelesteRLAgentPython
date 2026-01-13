import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from config import EnvironmentConfig, ObservationConfig


class CelesteFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN for grid input + MLP for static input.
    All layer sizes come from EnvironmentConfig and ObservationConfig.
    """

    def __init__(
        self,
        observation_space,
        grid_channels=ObservationConfig.CATEGORY_COUNT,
        grid_size=ObservationConfig.GRID_SIZE,
        static_size=ObservationConfig.STATIC_FEATURE_COUNT,
        cnn_channels=(32, 64),   
        mlp_hidden=EnvironmentConfig.LAYER_NEURONS
    ):
        super().__init__(observation_space, features_dim=1)

        c1, c2 = cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_size, grid_size)
            cnn_out_size = self.cnn(dummy).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(static_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )

        self._features_dim = cnn_out_size + mlp_hidden

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, observations):
        grid = observations["grid"]
        static = observations["static"]

        grid_features = self.cnn(grid)
        static_features = self.mlp(static)

        return torch.cat([grid_features, static_features], dim=1)

