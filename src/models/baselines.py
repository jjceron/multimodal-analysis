from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def _normalize_cov(x: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    cov = np.cov(x.reshape(x.shape[0], -1))
    n = cov.shape[0]
    cov_shrunk = (1 - shrinkage) * cov + shrinkage * np.eye(n) * np.trace(cov) / n
    return cov_shrunk / np.trace(cov_shrunk)


# ---------------------------------------------------------------------------
# CSP + LDA
# ---------------------------------------------------------------------------
class CSPLDA(nn.Module):
    _is_sklearn = True

    def __init__(
        self,
        n_channels: int = 128,
        n_classes: int = 2,
        n_components: int = 4,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_components = n_components
        self.filters_: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.lda: LinearDiscriminantAnalysis | None = None
        self.fitted = False

    def _compute_csp(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        covs = {}
        for cls in classes:
            X_cls = X[y == cls]
            covs[cls] = np.mean([_normalize_cov(x) for x in X_cls], axis=0)

        reg = 1e-6 * np.trace(covs[classes[0]] + covs[classes[1]]) / self.n_channels
        B = covs[classes[0]] + covs[classes[1]] + reg * np.eye(self.n_channels)
        eigvals, eigvecs = linalg.eigh(covs[classes[0]], B)
        idx = np.argsort(eigvals)[::-1]
        W = eigvecs[:, idx]
        n = self.n_components
        selected = np.concatenate([W[:, :n], W[:, -n:]], axis=1)
        return selected.T

    def _csp_features(self, X: np.ndarray) -> np.ndarray:
        projected = np.tensordot(self.filters_, X, axes=(1, 1))
        projected = projected.transpose(1, 0, 2)
        var = np.var(projected, axis=2)
        features = np.log(var / (var.sum(axis=1, keepdims=True) + 1e-10))
        return features

    def _collect(self, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for batch in loader:
            _, X, y = batch
            X_list.append(_to_numpy(X))
            y_list.append(_to_numpy(y))
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def fit(self, train_loader, device=None) -> None:
        X, y = self._collect(train_loader)
        self.filters_ = self._compute_csp(X, y)
        features = self._csp_features(X)
        features = self.scaler.fit_transform(features)
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(features, y)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if not self.fitted:
            logits = torch.zeros(x.size(0), self.n_classes)
            return logits, None
        X = _to_numpy(x)
        features = self._csp_features(X)
        features = self.scaler.transform(features)
        probs = self.lda.predict_proba(features)
        logits = torch.log(torch.tensor(probs, dtype=torch.float32) + 1e-10)
        return logits, None


# ---------------------------------------------------------------------------
# Riemannian MDM
# ---------------------------------------------------------------------------
class RiemannianMDM(nn.Module):
    _is_sklearn = True

    def __init__(
        self,
        n_channels: int = 128,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.mdm: nn.Module | None = None
        self.fitted = False

    def _compute_covs(self, X: np.ndarray) -> np.ndarray:
        from pyriemann.estimation import Covariances
        return Covariances(estimator='lwf').transform(X)

    def _collect(self, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for batch in loader:
            _, X, y = batch
            X_list.append(_to_numpy(X))
            y_list.append(_to_numpy(y))
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def fit(self, train_loader, device=None) -> None:
        from pyriemann.classification import MDM

        X, y = self._collect(train_loader)
        covs = self._compute_covs(X)
        self.mdm = MDM()
        self.mdm.fit(covs, y)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if not self.fitted:
            return torch.zeros(x.size(0), self.n_classes), None
        X = _to_numpy(x)
        covs = self._compute_covs(X)
        probs = self.mdm.predict_proba(covs)
        logits = torch.log(torch.tensor(probs, dtype=torch.float32) + 1e-10)
        return logits, None


# ---------------------------------------------------------------------------
# Band Power + SVM
# ---------------------------------------------------------------------------
class BandPowerSVM(nn.Module):
    _is_sklearn = True

    def __init__(
        self,
        n_channels: int = 128,
        n_classes: int = 2,
        window_sec: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sfreq = 250.0
        self.nperseg = min(256, int(window_sec * self.sfreq))
        self.pipeline: Pipeline | None = None
        self.fitted = False

    def _extract_band_power(self, X: np.ndarray) -> np.ndarray:
        B, C, T = X.shape
        f, psd = signal.welch(X.reshape(-1, T), fs=self.sfreq, nperseg=self.nperseg, axis=-1)
        psd = psd.reshape(B, C, -1)
        features = []
        for lo, hi in FREQ_BANDS.values():
            mask = (f >= lo) & (f <= hi)
            bp = np.trapezoid(psd[:, :, mask], f[mask], axis=2)
            features.append(bp)
        return np.concatenate(features, axis=1)

    def _collect(self, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for batch in loader:
            _, X, y = batch
            X_list.append(_to_numpy(X))
            y_list.append(_to_numpy(y))
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def fit(self, train_loader, device=None) -> None:
        X, y = self._collect(train_loader)
        features = self._extract_band_power(X)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced",
                        probability=True)),
        ])
        self.pipeline.fit(features, y)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if not self.fitted:
            return torch.zeros(x.size(0), self.n_classes), None
        X = _to_numpy(x)
        features = self._extract_band_power(X)
        probs = self.pipeline.predict_proba(features)
        logits = torch.log(torch.tensor(probs, dtype=torch.float32) + 1e-10)
        return logits, None
