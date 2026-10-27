#!/usr/bin/env python3
from __future__ import annotations

# region book:fnd.optimization-regularization.core-01
import numpy as np
from pydantic import BaseModel, Field


class RegularizationConfig(BaseModel):
    l2: float = Field(..., ge=0.0)


def l2_penalty(weights: np.ndarray, config: RegularizationConfig) -> float:
    return float(config.l2 * np.sum(weights**2))


# endregion book:fnd.optimization-regularization.core-01


# region book:fnd.optimization-regularization.example-01
cfg = RegularizationConfig(l2=0.01)
weights = np.array([0.5, -0.3, 0.1])
print({"l2_penalty": round(l2_penalty(weights, cfg), 4)})
# endregion book:fnd.optimization-regularization.example-01


# region book:fnd.transformers-foundation-models.recurrence-01 output
import numpy as np


def recurrent_step(
    x: np.ndarray,
    h: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return np.tanh(W_hh @ h + W_xh @ x + b)


def run_recurrence(
    xs: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b: np.ndarray,
    h0: np.ndarray,
) -> np.ndarray:
    h = h0.copy()  # Each sequence starts from its own state.
    states = []
    for x in xs:
        h = recurrent_step(x, h, W_xh, W_hh, b)
        states.append(h)
    return np.array(states).reshape(-1, len(h0))


W_hh, W_xh = np.array([[0.5]]), np.array([[1.0]])
b, h0 = np.array([0.0]), np.array([0.0])
paid, returned = 1.0, 0.5  # Given token codes, not learned embeddings.
print(
    f"w={W_hh.item():.1f}; h0={h0.item():.1f}; "
    f"paid={paid:.1f}; returned={returned:.1f}"
)
for label, codes in (
    ("paid, returned", (paid, returned)),
    ("returned, paid", (returned, paid)),
):
    states = run_recurrence(np.array(codes)[:, None], W_xh, W_hh, b, h0)
    print(
        f"{label}: states "
        + " ".join(f"{h:.3f}" for h in states[:, 0])
        + f"; code sum {sum(codes):.1f}"
    )
# endregion book:fnd.transformers-foundation-models.recurrence-01


# region book:fnd.transformers-foundation-models.bptt-01 output
def gradient_through_time(
    w: float, h0: float, n: int, tanh: bool = True
) -> tuple[float, float]:
    if n < 0:
        raise ValueError("Step count must be nonnegative.")
    if not tanh:
        return w**n, h0 * w**n
    h, product = h0, 1.0
    for _ in range(n):
        h = float(np.tanh(w * h))
        product *= w * (1 - h**2)
    return product, h


def central_difference(w: float, h0: float, n: int, eps: float) -> float:
    if eps <= 0:
        raise ValueError("Difference step must be positive.")
    above = gradient_through_time(w, h0 + eps, n)[1]
    below = gradient_through_time(w, h0 - eps, n)[1]
    return (above - below) / (2 * eps)


n, h0, eps = 10, 0.1, 1e-6
print(f"n={n}; h0={h0:.1f}; eps={eps:.6f}")
for w in (0.5, 1.0, 1.2):
    linear, _ = gradient_through_time(w, h0, n, tanh=False)
    product, _ = gradient_through_time(w, h0, n)
    difference = central_difference(w, h0, n, eps)
    print(
        f"w={w:.1f}: linear={linear:.6f}; tanh={product:.6f}; "
        f"central={difference:.6f}"
    )
for length in (5, 2048):
    print(
        f"tokens {length}: recurrent links {length - 1}, causal-attention "
        f"links 1"
    )
# endregion book:fnd.transformers-foundation-models.bptt-01


# region book:fnd.transformers-foundation-models.lstm-01 output
def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


def lstm_step(
    x: float, h: float, c: float,
    W: np.ndarray, U: np.ndarray, biases: np.ndarray,
) -> dict[str, float]:
    # Chung et al., sec. 3.1, without peepholes V_f, V_i, V_o.
    scores = W * x + U * h + biases  # forget, input, output, candidate
    f, i, o = sigmoid(scores[:3])
    candidate = float(np.tanh(scores[3]))
    cell = f * c + i * candidate
    state = o * np.tanh(cell)
    return dict(forget=f, input=i, output=o, candidate=candidate,
                cell=cell, state=state)


def gru_step(
    x: float, h: float,
    W: np.ndarray, U: np.ndarray, biases: np.ndarray,
) -> dict[str, float]:
    # Cho et al., sec. 2.3, eqs. 5-8: z retains the previous state.
    r = sigmoid(W[0] * x + U[0] * h + biases[0])
    z = sigmoid(W[1] * x + U[1] * h + biases[1])
    candidate = float(np.tanh(W[2] * x + U[2] * (r * h) + biases[2]))
    state = z * h + (1 - z) * candidate
    # Chung sec. 3.2, eq. 5 swaps z and 1-z; footnote 1's two
    # reset placements give the same number for one unit.
    return dict(reset=r, update=z, candidate=candidate, state=state)


def recurrent_weight_count(d: int, d_h: int, gates: int = 1) -> int:
    return gates * (d_h * (d + d_h) + d_h)  # One bias per unit per gate.


lstm_W, lstm_U = np.ones(4), np.full(4, 0.5)
lstm_biases = np.array([0.0, -0.5, 0.5, 0.0])
gru_W, gru_U = np.ones(3), np.full(3, 0.5)
gru_biases = np.array([0.0, -0.5, 0.0])
print("Constructed one-unit steps: x=1.0; h0=0.0; c0=0.0")
print("LSTM order: forget, input, output, candidate; no peepholes")
print(f"LSTM W_x={lstm_W.tolist()}; W_h={lstm_U.tolist()}")
print(f"LSTM biases={lstm_biases.tolist()}")
lstm_values = lstm_step(1.0, 0.0, 0.0, lstm_W, lstm_U, lstm_biases)
print("LSTM gates:", "; ".join(
    f"{k}={lstm_values[k]:.3f}" for k in ("forget", "input", "output")
))
print("LSTM values:", "; ".join(
    f"{k}={lstm_values[k]:.3f}" for k in ("candidate", "cell", "state")
))
print("GRU order: reset, update, candidate; update retains old state")
print(f"GRU W_x={gru_W.tolist()}; W_h={gru_U.tolist()}")
print(f"GRU biases={gru_biases.tolist()}")
gru_values = gru_step(1.0, 0.0, gru_W, gru_U, gru_biases)
print("GRU:", "; ".join(f"{k}={v:.3f}" for k, v in gru_values.items()))
print("Cell path only, forget held at 0.95; linear recurrence w=0.5")
cell_paths = np.array([0.95**t for t in range(1, 11)])
plain_paths = np.array([
    gradient_through_time(0.5, 0.0, t, tanh=False)[0]
    for t in range(1, 11)
])
for t, (cell, plain) in enumerate(zip(cell_paths, plain_paths), 1):
    print(f"path t={t}: cell={cell:.6f}; recurrence={plain:.6f}")
print("Weights including biases: d=512; d_h=512")
for label, gates in (("LSTM", 4), ("GRU", 3), ("recurrence", 1)):
    print(f"{label}: {recurrent_weight_count(512, 512, gates)}")
print("weight ratio LSTM:GRU:recurrence = 4:3:1")
# endregion book:fnd.transformers-foundation-models.lstm-01


# region book:fnd.transformers-foundation-models.rollout-01 output
def rollout(level: float, factor: float, steps: int, teacher_forced: bool) -> np.ndarray:
    if steps < 0:
        raise ValueError("Step count must be nonnegative.")
    current, rows = level, []
    for _ in range(steps):
        prediction = factor * current
        rows.append((current, prediction, prediction - level))
        current = level if teacher_forced else prediction
    return np.array(rows).reshape(-1, 3)  # input, prediction, signed error


level, factor, steps = 100.0, 1.05, 7
forced = rollout(level, factor, steps, teacher_forced=True)
free = rollout(level, factor, steps, teacher_forced=False)
print(f"level={level:.0f}; factor={factor:.2f}; steps={steps}")
print("step | truth prediction error | free input prediction error")
for step, (truth, own) in enumerate(zip(forced, free), 1):
    print(f"{step:4d} | {truth[1]:8.2f} {truth[2]:5.2f} | "
          f"{own[0]:8.2f} {own[1]:8.2f} {own[2]:5.2f}")
print(f"largest input: truth={forced[:, 0].max():.2f}; free={free[:, 0].max():.2f}")
print(f"day-{steps} / day-1 free error: {free[-1, 2] / free[0, 2]:.1f}")
# endregion book:fnd.transformers-foundation-models.rollout-01


# region book:fnd.transformers-foundation-models.core-01 output
import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    maximum = scores.max(axis=-1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("Each row needs a finite unmasked score.")
    exponentials = np.exp(scores - maximum)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    d_k = queries.shape[-1]
    scores = queries @ keys.T / np.sqrt(d_k)
    weights = softmax(
        scores if mask is None else scores + mask
    )  # Add 0 or -np.inf.
    return weights @ values, weights
# endregion book:fnd.transformers-foundation-models.core-01


# region book:fnd.transformers-foundation-models.example-01 output
queries = np.array([[1.0, 0.0]])
keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
values = np.array([[10.0], [0.0], [20.0]])
output, weights = scaled_dot_product_attention(queries, keys, values)
mask = np.array([[0.0, 0.0, -np.inf]])
masked_output, masked_weights = scaled_dot_product_attention(
    queries, keys, values, mask
)
print("unmasked weights:", " ".join(f"{weight:.3f}" for weight in weights[0]))
print(f"unmasked output: {output.item():.3f}")
print(
    "masked weights:",
    " ".join(f"{weight:.3f}" for weight in masked_weights[0]),
)
print(f"masked output: {masked_output.item():.3f}")
# endregion book:fnd.transformers-foundation-models.example-01


# region book:fnd.transformers-foundation-models.scale-01 output
rng = np.random.default_rng(0)
for d_k in (4, 64, 512):
    queries = rng.standard_normal((10_000, d_k))
    keys = rng.standard_normal((10_000, d_k))
    scores = np.sum(queries * keys, axis=1)
    unscaled_variance = float(scores.var())
    scaled_variance = float((scores / np.sqrt(d_k)).var())
    print(
        f"d_k={d_k}: variance={unscaled_variance:.3f}, "
        f"scaled={scaled_variance:.3f}"
    )
# endregion book:fnd.transformers-foundation-models.scale-01


# region book:fnd.transformers-foundation-models.positions-01 output
n, d = 3, 4
positions = np.zeros((n, d))
angles = np.arange(n)[:, None] / 10_000.0 ** (np.arange(0, d, 2) / d)
positions[:, 0::2] = np.sin(angles)
positions[:, 1::2] = np.cos(angles)
token_vectors = np.eye(d)[:n]
permutation = [2, 1, 0]
print(
    "position 1:", " ".join(f"{component:.3f}" for component in positions[1])
)
for label, encoding in (
    ("without", np.zeros_like(positions)),
    ("with", positions),
):
    original = token_vectors + encoding
    reordered = token_vectors[permutation] + encoding
    output, _ = scaled_dot_product_attention(original, original, original)
    reordered_output, _ = scaled_dot_product_attention(
        reordered, reordered, reordered
    )
    equivariant = np.allclose(reordered_output, output[permutation])
    print(f"equivariant {label} positions: {equivariant}")
# endregion book:fnd.transformers-foundation-models.positions-01


# region book:fnd.transformers-foundation-models.block-01 output
def layer_norm(token_vectors: np.ndarray) -> np.ndarray:
    centered = token_vectors - token_vectors.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(token_vectors.var(axis=-1, keepdims=True) + 1e-5)


def block_weight_count(d: int, d_ff: int) -> int:
    return (
        4 * d**2 + 2 * d * d_ff
    )  # Projection matrices; layer norm uses fixed gain and bias.


rng = np.random.default_rng(0)
n, d, d_ff = 5, 8, 16
token_vectors = rng.standard_normal((n, d))
query_projection, key_projection, value_projection, output_projection = (
    rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(4)
)
feed_forward_in = rng.standard_normal((d, d_ff)) / np.sqrt(d)
feed_forward_out = rng.standard_normal((d_ff, d)) / np.sqrt(d_ff)
queries = token_vectors @ query_projection
keys = token_vectors @ key_projection
values = token_vectors @ value_projection
attention_output, _ = scaled_dot_product_attention(queries, keys, values)
attention_output = attention_output @ output_projection
attention_residual = layer_norm(token_vectors + attention_output)
hidden = np.maximum(0.0, attention_residual @ feed_forward_in)
feed_forward_output = hidden @ feed_forward_out
block_output = layer_norm(attention_residual + feed_forward_output)
print(f"input: {token_vectors.shape}")
print(f"attention: {attention_output.shape}")
print(f"attention add/norm: {attention_residual.shape}")
print(f"feed-forward hidden: {hidden.shape}")
print(f"feed-forward output: {feed_forward_output.shape}")
print(f"block output: {block_output.shape}")
print(f"matrix weights (d=512, d_ff=2048): {block_weight_count(512, 2048)}")
# endregion book:fnd.transformers-foundation-models.block-01


# region book:fnd.transformers-foundation-models.loss-01 output
tokens = ("The", "sky", "is", "blue")
probabilities = np.array([0.5, 0.25, 0.5, 0.125])
losses = -np.log(probabilities)
for token, loss in zip(tokens, losses):
    print(f"{token}: {loss:.3f}")
mean_loss = float(losses.mean())
perplexity = float(np.exp(mean_loss))
print(f"mean loss: {mean_loss:.3f}")
print(f"perplexity: {perplexity:.3f}")
# endregion book:fnd.transformers-foundation-models.loss-01


# region book:fnd.transformers-foundation-models.decoder-01 output

def decoder_forward(
    token_ids: np.ndarray,
    embedding_table: np.ndarray,
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    embeddings = embedding_table[token_ids]
    n, d = embeddings.shape
    positions = np.zeros((n, d))
    angles = np.arange(n)[:, None] / 10_000.0 ** (np.arange(0, d, 2) / d)
    positions[:, 0::2], positions[:, 1::2] = np.sin(angles), np.cos(angles)
    vectors = embeddings + positions
    mask = np.where(
        np.arange(n)[None, :] > np.arange(n)[:, None], -np.inf, 0.0
    )
    attended, attention_weights = scaled_dot_product_attention(
        vectors @ weights["query"],
        vectors @ weights["key"],
        vectors @ weights["value"],
        mask,
    )
    attention = attended @ weights["output"]
    residual = layer_norm(vectors + attention)
    hidden = np.maximum(0.0, residual @ weights["feed_forward_in"])
    feed_forward = hidden @ weights["feed_forward_out"]
    return {
        "embeddings": embeddings,
        "positions": positions,
        "input": vectors,
        "attention weights": attention_weights,
        "attention": attention,
        "attention add/norm": residual,
        "feed-forward hidden": hidden,
        "feed-forward output": feed_forward,
        "block output": layer_norm(residual + feed_forward),
    }


vocabulary = ("(start)", "The", "sky", "is", "blue", "cloudy", "(end)")
token_ids, target_ids = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])
rng = np.random.default_rng(7)
d, d_ff = 8, 16
embedding_table = rng.standard_normal((len(vocabulary), d)) / np.sqrt(d)
decoder_weights = {
    name: rng.standard_normal((d, d)) / np.sqrt(d)
    for name in ("query", "key", "value", "output")
}
decoder_weights["feed_forward_in"] = rng.standard_normal((d, d_ff)) / np.sqrt(
    d
)
decoder_weights["feed_forward_out"] = rng.standard_normal((d_ff, d)) / np.sqrt(
    d_ff
)
output_head = rng.standard_normal((d, len(vocabulary))) / np.sqrt(d)
frozen_weights = {
    "embedding": embedding_table.copy(),
    **{name: value.copy() for name, value in decoder_weights.items()},
}
decoder_trace = decoder_forward(token_ids, embedding_table, decoder_weights)
decoder_hidden = decoder_trace["block output"]
decoder_logits = decoder_hidden @ output_head
decoder_probabilities = softmax(decoder_logits)
target_probabilities = decoder_probabilities[
    np.arange(len(token_ids)), target_ids
]
decoder_loss = float(-np.log(target_probabilities).mean())
decoder_perplexity = float(np.exp(decoder_loss))
print(
    "seed 7; vocabulary:",
    ", ".join(f"{i}={token}" for i, token in enumerate(vocabulary)),
)
print(
    "input ids:",
    token_ids.tolist(),
    "; shifted target ids:",
    target_ids.tolist(),
)
for name, values in decoder_trace.items():
    print(f"{name}: {values.shape}")
print(
    f"vocabulary scores: {decoder_logits.shape}; probabilities: "
    f"{decoder_probabilities.shape}"
)
for token_id, embedding, position in zip(
    token_ids, decoder_trace["embeddings"], decoder_trace["positions"]
):
    print(
        f"{vocabulary[token_id]}: embedding[0:4] "
        + " ".join(f"{v:.3f}" for v in embedding[:4])
        + ";\n  position[0:4] "
        + " ".join(f"{v:.3f}" for v in position[:4])
    )
print(
    "last attention weights:",
    " ".join(f"{v:.3f}" for v in decoder_trace["attention weights"][-1]),
)
print(
    "last block row:", " ".join(f"{v:.3f}" for v in decoder_hidden[-1])
)
for target_id, probability in zip(target_ids, target_probabilities):
    print(
        f"target {vocabulary[target_id]}: probability {probability:.6f}; loss "
        f"{-np.log(probability):.6f}"
    )
print(
    f"mean loss: {decoder_loss:.6f}; perplexity: {decoder_perplexity:.6f}"
)
print(
    f"geometric mean target probability: "
    f"{np.prod(target_probabilities) ** (1 / len(token_ids)):.6f}"
)
for index in np.argsort(-decoder_probabilities[-1], kind="stable")[:5]:
    print(
        f"next {vocabulary[index]}: score {decoder_logits[-1, index]:.6f}; "
        f"probability {decoder_probabilities[-1, index]:.6f}"
    )
appended_ids = np.r_[token_ids, int(np.argmax(decoder_probabilities[-1]))]
print(
    "greedy appended context:", " ".join(vocabulary[i] for i in appended_ids)
)
head_before = output_head.copy()
score_gradient = decoder_probabilities.copy()
score_gradient[np.arange(len(token_ids)), target_ids] -= (
    1  # p - y at each position.
)
head_gradient = decoder_hidden.T @ score_gradient / len(token_ids)
head_learning_rate = 0.1
output_head = output_head - head_learning_rate * head_gradient
updated_probabilities = softmax(decoder_hidden @ output_head)
updated_decoder_loss = float(
    -np.log(
        updated_probabilities[np.arange(len(token_ids)), target_ids]
    ).mean()
)
print(
    f"head-only step: learning rate {head_learning_rate:g}; loss "
    f"{decoder_loss:.6f} -> {updated_decoder_loss:.6f}"
)
print(
    "changed weights: W_U only; no other weight moved;\n  random decoder, no "
    "language model trained"
)
# endregion book:fnd.transformers-foundation-models.decoder-01


# region book:fnd.transformers-foundation-models.contrastive-01 output
def contrastive_comparison(
    images: np.ndarray, captions: np.ndarray, tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    cosines = images @ captions.T  # Inputs here are unit vectors.
    row_weights = softmax(cosines / tau)
    column_weights = softmax(cosines.T / tau).T
    image_loss = float(-np.log(np.diag(row_weights)).mean())
    caption_loss = float(-np.log(np.diag(column_weights)).mean())
    losses = (image_loss, caption_loss, (image_loss + caption_loss) / 2)
    return cosines, row_weights, column_weights, losses


image_vectors = np.eye(3)
caption_vectors = np.array([[0.8, 0.6, 0], [0, 0.8, 0.6], [0.6, 0, 0.8]])
temperature = 0.1
print("Constructed unit vectors; no model outputs; fixed illustrative tau=0.1")
for label, vectors in (("image", image_vectors), ("caption", caption_vectors)):
    for index, vector in enumerate(vectors, 1):
        print(f"{label} {index}:", " ".join(f"{v:.1f}" for v in vector))
print("Swap captions 2 and 3; keep the diagonal as the target pairing.")
contrastive_results = {}
for label, captions in (
    ("matched", caption_vectors),
    ("swapped", caption_vectors[[0, 2, 1]]),
):
    result = contrastive_comparison(image_vectors, captions, temperature)
    contrastive_results[label] = result
    cosines, row_weights, column_weights, losses = result
    for kind, matrix in (("cosine", cosines), ("row softmax", row_weights)):
        for index, row in enumerate(matrix, 1):
            print(f"{label} {kind} row {index}:",
                  " ".join(f"{v:.6f}" for v in row))
    print(f"{label} image-to-caption loss: {losses[0]:.6f}")
    print(f"{label} caption-to-image loss: {losses[1]:.6f}")
    print(f"{label} symmetric loss: {losses[2]:.6f}")
# endregion book:fnd.transformers-foundation-models.contrastive-01


# region book:fnd.transformers-foundation-models.icl-01 output
W_Q, W_K, W_V = np.eye(2), np.eye(2), np.ones((1, 1))
icl_projections = (W_Q, W_K, W_V)
icl_query = np.array([[1.0, 0.0]])
contexts = (
    ("A", np.array([[1., 0.], [0., 1.], [1., 1.]]), [0., 10., 20.]),
    ("B", np.array([[0., 1.], [1., 0.], [0., 2.]]), [10., 0., 40.]),
)
print("Constructed one-head reading mechanism, "
      "not evidence about a trained model.")
print("Fixed W_Q=W_K=identity of width 2; W_V=[1]; query=1.0 0.0")
icl_results = []
for label, inputs, labels in contexts:
    values = np.array(labels)[:, None] @ W_V
    prediction, weights = scaled_dot_product_attention(
        icl_query @ W_Q, inputs @ W_K, values
    )
    fingerprint = float(sum(weight.sum() for weight in icl_projections))
    icl_results.append((weights[0].copy(), prediction.item(), fingerprint))
    for index, (x, value) in enumerate(zip(inputs, labels), 1):
        print(f"context {label} demo {index}: input={x[0]:.1f} {x[1]:.1f}; "
              f"label={value:.1f}")
    print(f"context {label} weights:",
          " ".join(f"{v:.6f}" for v in weights[0]))
    print(f"context {label} prediction: {prediction.item():.6f}")
    print(f"context {label} weight fingerprint (sum): {fingerprint:.3f}")
print("No weight changed; keys come from inputs and values carry labels.")
# endregion book:fnd.transformers-foundation-models.icl-01


# region book:fnd.transformers-foundation-models.rollout-02 output
print("Random decoder, seed 7; nothing in this rollout is trained.")
# decoder-01 saved this head before its illustrative gradient step.
generation_ids = token_ids.copy()
generation_trace = []
for step in range(1, 4):
    trace = decoder_forward(generation_ids, embedding_table, decoder_weights)
    probabilities = softmax(trace["block output"][-1:] @ head_before)[0]
    action = int(probabilities.argmax())
    appended = np.r_[generation_ids, action]
    generation_trace.append(
        (generation_ids.copy(), action, float(probabilities[action]), appended)
    )
    print(f"step {step} state:",
          " ".join(vocabulary[i] for i in generation_ids))
    print(f"step {step} action: {vocabulary[action]}; "
          f"probability={probabilities[action]:.6f}")
    print(f"step {step} appended:", " ".join(vocabulary[i] for i in appended))
    generation_ids = appended
# endregion book:fnd.transformers-foundation-models.rollout-02


# region book:fnd.transformers-foundation-models.sota-01 output
def kv_cache_bytes(
    n_blocks: int, n_heads: int, d_k: int, n: int, bytes_per_number: int
) -> int:
    if n < 0 or min(n_blocks, n_heads, d_k, bytes_per_number) <= 0:
        raise ValueError(
            "Dimensions and byte width must be positive; token count "
            "nonnegative."
        )
    return 2 * n_blocks * n_heads * d_k * n * bytes_per_number


def attention_score_counts(n: int) -> tuple[int, int]:
    """Count scores per head per block, growing an empty prefix to n tokens."""
    if n < 0:
        raise ValueError("Token count must be nonnegative.")
    cached = n * (n + 1) // 2
    uncached = n * (n + 1) * (2 * n + 1) // 6
    return cached, uncached


n = 2048
cache_bytes = kv_cache_bytes(
    n_blocks=24, n_heads=8, d_k=64, n=n, bytes_per_number=2
)
cached_scores, uncached_scores = attention_score_counts(n)
print(f"cache bytes: {cache_bytes}")
print(f"cache MiB: {cache_bytes / 2**20:.3f}")
print(f"cached scores: {cached_scores}")
print(f"uncached scores: {uncached_scores}")
# endregion book:fnd.transformers-foundation-models.sota-01


# region book:fnd.transformers-foundation-models.serving-01 output
print("Counts use multiplications; "
      "hardware often counts a multiply and an add")
print("as two operations. Weight traffic only; cache attention is omitted.")
serving_weight_count = 24 * block_weight_count(512, 2048)
memory_limit = 2**30
print("Configuration: L=24; H=8; d_k=d_v=64; d=512; d_ff=2048; n=2048")
print(f"matrix weights: per block={block_weight_count(512, 2048)}; "
      f"all blocks={serving_weight_count}")
print(f"Illustrative memory for weights and cache: "
      f"{memory_limit} bytes (1 GiB)")
print("Only matrix weights and cache count toward this illustrative budget.")
serving_rows = []
for b in (2, 1, 0.5):
    weight_bytes = int(serving_weight_count * b)
    cache_token = int(kv_cache_bytes(24, 8, 64, 1, b))
    cache_request = int(kv_cache_bytes(24, 8, 64, 2048, b))
    left = memory_limit - weight_bytes
    n_req = left // cache_request
    serving_rows.append(
        (b, weight_bytes, cache_token, cache_request, left, n_req)
    )
    print(f"b={b:g}: weights={weight_bytes} bytes; "
          f"weights MiB={weight_bytes / 2**20:.3f}")
    print(f"b={b:g}: cache/token={cache_token} bytes; "
          f"cache/request={cache_request} bytes; "
          f"MiB={cache_request / 2**20:.3f}")
    print(f"b={b:g}: bytes left={left}; full-window requests n_req={n_req}")
prefill_intensity = 2048 * serving_weight_count / (2 * serving_weight_count)
print("prefill tokens=2048; b=2; "
      f"multiplications/weight byte={prefill_intensity:.3f}")
batch_table = [(n_req, n_req / 2) for n_req in (1, 2, 4, 8, 16, 32, 64)]
for n_req, intensity in batch_table:
    print(f"decode n_req={n_req}; b=2; "
          f"multiplications/weight byte={intensity:.3f}")
print("Each request also reads its own cache every step: 49152 x its tokens")
print("bytes at b=2, in addition to the shared weight read.")
print("Constructed independent calls: slow chance=0.01 per call.")
tail_table = [(k, 1 - 0.99**k) for k in (1, 2, 5, 10, 20, 50, 100, 200, 500)]
for k, chance in tail_table:
    print(f"tail k={k}: chance={chance:.6f}; percent={100 * chance:.3f}")
print("Constructed latencies: fast=1 unit; slow=10 units; not measured.")
print("Simulation: runs=10000; seed=19; tolerance=0.02 in share.")
latency_rng = np.random.default_rng(19)
tail_simulation = []
for k in (10, 100):
    constructed_latencies = np.where(
        latency_rng.random((10_000, k)) < 0.01, 10, 1
    )
    share = float((constructed_latencies == 10).any(axis=1).mean())
    tail_simulation.append((k, share))
    print(f"simulation k={k}: slow-run share={share:.6f}; "
          f"formula={1 - 0.99**k:.6f}")
# endregion book:fnd.transformers-foundation-models.serving-01


# region book:fnd.transformers-foundation-models.keys-07 output
print(f"gradient factor (w=0.8, n=20): {gradient_through_time(0.8, 0.1, 20, tanh=False)[0]:.4f}")
level, factor, steps = 50.0, 1.02, 10
exercise_rollout = rollout(level, factor, steps, teacher_forced=False)
print(f"level={level:.0f}; factor={factor:.2f}; steps={steps}")
print(f"one-step error: {exercise_rollout[0, 2]:.2f}")
print(f"ten-step free-running error: {exercise_rollout[-1, 2]:.2f}")
# endregion book:fnd.transformers-foundation-models.keys-07


# region book:fnd.transformers-foundation-models.keys-12 output
print("LSTM exercise, constructed givens: x=0.5; h0=0.0; c0=0.0")
exercise_W, exercise_U = np.ones(4), np.full(4, 0.25)
exercise_biases = np.array([0.0, -0.5, 0.0, 0.0])
print("order: forget, input, output, candidate; no peepholes")
print(f"W_x={exercise_W.tolist()}; W_h={exercise_U.tolist()}")
print(f"biases={exercise_biases.tolist()}")
print("cell-path givens: forget held at 0.9; steps=20")
exercise_lstm = lstm_step(
    0.5, 0.0, 0.0, exercise_W, exercise_U, exercise_biases
)
exercise_cell_path = 0.9**20
print("LSTM key gates:", "; ".join(
    f"{k}={exercise_lstm[k]:.3f}" for k in ("forget", "input", "output")
))
print("LSTM key values:", "; ".join(
    f"{k}={exercise_lstm[k]:.3f}" for k in ("candidate", "cell", "state")
))
print(f"cell-path key: {exercise_cell_path:.6f}")

print("In-context exercise, constructed givens: d_k=1; query=0.5")
print("W_Q=W_K=W_V=[1]; keys equal inputs; values equal labels")
exercise_inputs = np.array([[0.0], [1.0], [2.0]])
print("demonstration inputs: 0.0 1.0 2.0")
print("first labels: 2.0 4.0 8.0")
print("second labels: 2.0 10.0 8.0; only the middle label changed")
exercise_icl_weights, exercise_icl_predictions = [], []
for label, labels in (("first", [2., 4., 8.]), ("second", [2., 10., 8.])):
    prediction, weights = scaled_dot_product_attention(
        np.array([[0.5]]), exercise_inputs, np.array(labels)[:, None]
    )
    exercise_icl_weights.append(weights[0].copy())
    exercise_icl_predictions.append(prediction.item())
    print(f"{label} key weights:", " ".join(f"{v:.6f}" for v in weights[0]))
    print(f"{label} key prediction: {prediction.item():.6f}")
print("Attention weights and projection weights are unchanged.")

print("Serving exercise, illustrative givens:")
print("L=12; H=12; d_k=d_v=64; d=768; d_ff=3072; tokens/request=4096")
print("weight b=2 and 1; cache b=2 fixed; memory=2147483648 bytes (2 GiB)")
print("tail givens: calls=20; independent slow chance=1/50")
exercise_weight_count = 12 * block_weight_count(768, 3072)
exercise_cache = kv_cache_bytes(12, 12, 64, 4096, 2)
print(f"key matrix weights: {exercise_weight_count}")
print(f"key cache/request: {exercise_cache} bytes; "
      f"MiB={exercise_cache / 2**20:.3f}")
exercise_serving_rows = []
for b in (2, 1):
    weight_bytes = exercise_weight_count * b
    left = 2**31 - weight_bytes
    n_req = left // exercise_cache
    exercise_serving_rows.append(
        (b, weight_bytes, exercise_cache, left, n_req)
    )
    print(f"key b={b}: weights={weight_bytes} bytes; "
          f"MiB={weight_bytes / 2**20:.3f}; bytes left={left}; n_req={n_req}")
exercise_tail = 1 - (1 - 1 / 50)**20
print(f"key tail chance: {exercise_tail:.6f}; "
      f"percent={100 * exercise_tail:.3f}")
# endregion book:fnd.transformers-foundation-models.keys-12


# region book:fnd.transformers-foundation-models.frameworks-01
import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# Optional training harness: use Lightning if available.
try:
    import lightning as L
except ModuleNotFoundError as exc:
    if exc.name != "lightning":
        raise
else:

    class TinyLightningModule(L.LightningModule):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.block = TinyBlock(dim)

        def training_step(self, batch, batch_idx: int) -> torch.Tensor:
            x, y = batch
            preds = self.block(x)
            return nn.functional.mse_loss(preds, y)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
# endregion book:fnd.transformers-foundation-models.frameworks-01
