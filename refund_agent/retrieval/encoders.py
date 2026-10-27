"""Small NumPy encoders; examples are rows, and all fitted state is explicit."""

import re

import numpy as np


def words(text: str) -> list[str]:
    # Keep changed_mind and not_delivered as words; split other punctuation.
    return re.findall(r"\w+", text.lower())


def cosine(u, v) -> float:
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    length = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / length) if length else 0.0


class CountEncoder:
    def __init__(self, rarity: bool = False):
        self.rarity = rarity

    def fit(self, texts: list[str]):
        if not texts:
            raise ValueError("fit needs texts")
        self.vocabulary = sorted({word for text in texts for word in words(text)})
        self.position = {word: i for i, word in enumerate(self.vocabulary)}
        self.weights = np.ones(len(self.vocabulary))
        if self.rarity:
            df = (self.transform(texts) > 0).sum(axis=0)
            self.weights = np.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def has_terms(self, text: str) -> bool:
        return any(word in self.position for word in words(text))

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), len(self.vocabulary)))
        for row, text in enumerate(texts):
            for word in words(text):
                if word in self.position:
                    x[row, self.position[word]] += 1
        return x * self.weights

    def manifest(self) -> dict:
        return {
            "kind": "counts",
            "rarity": self.rarity,
            "vocabulary": self.vocabulary,
            "weights": self.weights.tolist(),
        }


class LinearAutoencoder:
    def __init__(self, width: int):
        self.width = width

    def fit(self, x, method="closed_form", seed=7, steps=150000, step_size=0.06):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all() or not 0 < self.width <= min(x.shape):
            raise ValueError("finite matrix and a width between 1 and min(n, d_in) required")
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        self.eigenvalues = singular**2 / len(x)
        if method == "closed_form":
            self.weights = vt[: self.width].T.copy()
            # SVD signs are arbitrary; give each column a stable sign for printing.
            for column in self.weights.T:
                if column[np.argmax(np.abs(column))] < 0:
                    column *= -1
            self.decoder = self.weights.T.copy()
        elif method == "gradient_descent":
            rng = np.random.default_rng(seed)
            self.weights = rng.normal(0, 0.1, (x.shape[1], self.width))
            self.decoder = rng.normal(0, 0.1, (self.width, x.shape[1]))
            for _ in range(steps):
                h = centered @ self.weights
                residual = h @ self.decoder - centered
                grad_w = 2 * centered.T @ residual @ self.decoder.T / len(x)
                grad_d = 2 * h.T @ residual / len(x)
                self.weights -= step_size * grad_w
                self.decoder -= step_size * grad_d
        else:
            raise ValueError("unknown fitting method")
        return self

    def transform(self, x) -> np.ndarray:
        return (np.asarray(x) - self.mean) @ self.weights

    def reconstruct(self, h) -> np.ndarray:
        return np.asarray(h) @ self.decoder + self.mean

    def error(self, x) -> float:
        residual = np.asarray(x) - self.reconstruct(self.transform(x))
        return float(np.mean(np.sum(residual**2, axis=1)))

    def manifest(self) -> dict:
        return {
            "kind": "linear_autoencoder",
            "width": self.width,
            "mean": self.mean.tolist(),
            "weights": self.weights.tolist(),
            "decoder": self.decoder.tolist(),
        }


class LinearTextEncoder:
    """Apply a fitted count encoder and its centered linear code to the same text."""

    def __init__(self, counts: CountEncoder, autoencoder: LinearAutoencoder):
        self.counts, self.autoencoder = counts, autoencoder

    def has_terms(self, text: str) -> bool:
        return self.counts.has_terms(text)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.autoencoder.transform(self.counts.transform(texts))

    def manifest(self) -> dict:
        return {"counts": self.counts.manifest(), "autoencoder": self.autoencoder.manifest()}


class PairEncoder:
    """A linear text map fitted on question/passage pairs, using cosine InfoNCE.

    The positive is paired with its question; other distinct passages are negatives.
    Repeated positive passages are deduplicated, never treated as false negatives.
    This optional implementation awaits alignment with chapter 6's forthcoming loss;
    the chapter 11 comparison therefore does not print a pair-trained row yet.
    """

    def __init__(self, width=2, temperature=0.2):
        if width < 1 or temperature <= 0:
            raise ValueError("positive width and temperature required")
        self.width, self.temperature = width, temperature

    def fit(self, pairs, seed=7, steps=1500, step_size=0.1):
        if not pairs:
            raise ValueError("training pairs required")
        passages = sorted({passage for _, passage in pairs})
        if len(passages) < 2:
            raise ValueError("contrastive training needs at least two distinct passages")
        self.counts = CountEncoder().fit([q for q, _ in pairs] + passages)
        q = self.counts.transform([q for q, _ in pairs])
        p = self.counts.transform(passages)
        targets = np.array([passages.index(passage) for _, passage in pairs])
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, (q.shape[1], self.width))
        self.losses = []
        for _ in range(steps):
            loss, gradient = self.loss_gradient(q, p, targets)
            self.losses.append(loss)
            self.weights -= step_size * gradient
        return self

    def loss_gradient(self, q, p, targets):
        qh, ph = q @ self.weights, p @ self.weights
        qnorm = np.maximum(np.linalg.norm(qh, axis=1, keepdims=True), 1e-12)
        pnorm = np.maximum(np.linalg.norm(ph, axis=1, keepdims=True), 1e-12)
        u, v = qh / qnorm, ph / pnorm
        scores = u @ v.T / self.temperature
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        loss = float(np.mean(np.log(exp.sum(axis=1)) - shifted[np.arange(len(q)), targets]))
        gradient = probabilities.copy()
        gradient[np.arange(len(q)), targets] -= 1
        gradient /= len(q) * self.temperature
        du, dv = gradient @ v, gradient.T @ u
        dqh = (du - u * np.sum(du * u, axis=1, keepdims=True)) / qnorm
        dph = (dv - v * np.sum(dv * v, axis=1, keepdims=True)) / pnorm
        return loss, q.T @ dqh + p.T @ dph

    def has_terms(self, text):
        return self.counts.has_terms(text)

    def transform(self, texts):
        return self.counts.transform(texts) @ self.weights

    def manifest(self):
        return {
            "kind": "pair_encoder",
            "loss": "cosine_info_nce",
            "temperature": self.temperature,
            "counts": self.counts.manifest(),
            "weights": self.weights.tolist(),
        }
