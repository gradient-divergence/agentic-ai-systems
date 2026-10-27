"""Seeded, offline outputs for chapter 13, generative models and simulation.

The printed regions run in chapter order with no intervening setup. Importing
does no fitting, printing or writing. Scenario fitting uses only the failure-free
training window; its proposed decisions never become the suite's expected labels.
The linear Gaussian VAE uses exact expectations and gradients, not sampled losses.
"""

# region book:fnd.generative-models.sample-01 output
import numpy as np


def sampling_distribution(scores, rule):
    scores = np.asarray(scores, dtype=float)
    name, tau = rule["name"], rule.get("tau", 1)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("a finite nonempty vector of scores is required")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be positive and finite")
    p = np.exp((scores - scores.max()) / tau)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    if name == "greedy":
        keep = order[:1]
    elif name == "top-k":
        k = rule["k"]
        if type(k) is not int or not 1 <= k <= len(p):
            raise ValueError("k must fit the vocabulary")
        keep = order[:k]
    elif name == "nucleus":
        mass = rule["p"]
        if not 0 < mass <= 1:
            raise ValueError("nucleus mass must be in (0, 1]")
        keep = order[
            : np.searchsorted(np.cumsum(p[order]), mass, side="left") + 1
        ]
    elif name == "temperature":
        keep = order
    else:
        raise ValueError("unknown sampling rule")
    result = np.zeros_like(p)
    result[keep] = p[keep] / p[keep].sum()
    return result


if __name__ == "__main__":
    scores = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
    print("tokens: deploy certificate DNS disk network")
    print("scores:", " ".join(f"{x:.1f}" for x in scores))
    rules = [("greedy", {"name": "greedy"})]
    rules += [
        (f"temperature {tau:g}", {"name": "temperature", "tau": tau})
        for tau in (0.5, 1, 2)
    ]
    rules += [
        ("top-2", {"name": "top-k", "k": 2}),
        ("nucleus 0.8", {"name": "nucleus", "p": 0.8}),
    ]
    for name, rule in rules:
        p = sampling_distribution(scores, rule)
        print(
            f"{name}: {' '.join(f'{v:.3f}' for v in p)}; certificate "
            f"{p[1]:.3f}"
        )
    for tau in (0.5, 1, 2):
        p = sampling_distribution(
            scores, {"name": "nucleus", "p": 0.8, "tau": tau}
        )
        print(
            f"nucleus 0.8 at temperature {tau:g}: {np.count_nonzero(p)} tokens"
        )
# endregion book:fnd.generative-models.sample-01


# region book:fnd.generative-models.likelihood-01 output
from datetime import datetime, timezone

from refund_agent.scenarios.checks import policy_label
from refund_agent.scenarios.materialize import to_records
from refund_agent.scenarios.model import PARENTS, ScenarioModel, training_data
from refund_agent.scenarios.tokens import FIELDS, VALUES, valid_cases

if __name__ == "__main__":
    model = ScenarioModel.fit()
    at = datetime(2026, 9, 15, tzinfo=timezone.utc)
    data = training_data()
    candidates = [
        (*case, policy_label(to_records(case, at, 7), data).outcome.value)
        for case in valid_cases()
    ]
    common, rare = (
        max(candidates, key=model.log_prob),
        min(candidates, key=model.log_prob),
    )
    print(
        f"training: {len(model.training_tokens)} failure-free requests; "
        f"{len(model.training_cases)} distinct cases"
    )
    print(
        f"training dates: {model.training_manifest['first']} to "
        f"{model.training_manifest['last']}"
    )
    for name, tokens in (("common", common), ("rare", rare)):
        print(f"{name}: {tokens}")
        context = {}
        for t, value in enumerate(tokens):
            key = tuple(context[FIELDS[i]] for i in PARENTS[t])
            counts = model.counts[t].get(key, [0] * len(VALUES[FIELDS[t]]))
            index = VALUES[FIELDS[t]].index(value)
            p = model.conditional(t, context)[index]
            print(
                f"  {FIELDS[t]}={value}: "
                f"({counts[index]}+1)/({sum(counts)}+{len(counts)}) = "
                f"{p:.6f}; log {np.log(p):.6f}"
            )
            context[FIELDS[t]] = value
        print(
            f"  total log-probability {model.log_prob(tokens):.6f}; "
            f"probability {np.exp(model.log_prob(tokens)):.9f}"
        )
    counted = np.array([4, 1, 1]) / 6
    tempered = np.sqrt(counted) / np.sqrt(counted).sum()
    print(
        "three denials, ordered deny approve refer:",
        " ".join(f"{p:.3f}" for p in counted),
    )
    print("at temperature 2:", " ".join(f"{p:.3f}" for p in tempered))
# endregion book:fnd.generative-models.likelihood-01


# region book:fnd.generative-models.elbo-01 output
def gaussian_kl(mu, variance):
    mu, variance = (
        np.asarray(mu, dtype=float),
        np.asarray(variance, dtype=float),
    )
    if (
        not np.isfinite(mu).all()
        or not np.isfinite(variance).all()
        or np.any(variance <= 0)
    ):
        raise ValueError("finite means and positive finite variances required")
    return float(0.5 * np.sum(mu**2 + variance - 1 - np.log(variance)))


def evidence_lower_bound(expected_log_likelihood, mu, variance):
    return float(expected_log_likelihood - gaussian_kl(mu, variance))


if __name__ == "__main__":
    x, noise_variance = 2.0, 1.0
    encoder_mean, encoder_variance = 1.0, 1.0
    posterior_variance = 1 / (1 + 1 / noise_variance)
    posterior_mean = posterior_variance * x / noise_variance
    expected_log_likelihood = -0.5 * (
        np.log(2 * np.pi * noise_variance)
        + ((x - encoder_mean) ** 2 + encoder_variance) / noise_variance
    )
    worked_bound = evidence_lower_bound(
        expected_log_likelihood, [encoder_mean], [encoder_variance]
    )
    marginal_variance = 1 + noise_variance
    worked_log_likelihood = -0.5 * (
        np.log(2 * np.pi * marginal_variance) + x**2 / marginal_variance
    )
    posterior_kl = 0.5 * (
        (encoder_variance + (encoder_mean - posterior_mean) ** 2)
        / posterior_variance
        - 1
        + np.log(posterior_variance / encoder_variance)
    )
    print(f"x {x:g}; prior N(0, 1); decoder N(h, {noise_variance:g})")
    print(
        f"posterior N({posterior_mean:g}, {posterior_variance:g}); encoder "
        f"N({encoder_mean:g}, {encoder_variance:g})"
    )
    print(f"KL N(1, 1) || N(0, 1): {gaussian_kl([1], [1]):.3f}")
    print(f"expected log-likelihood: {expected_log_likelihood:.3f}")
    print(f"evidence lower bound: {worked_bound:.3f}")
    print(
        f"log p(x): {worked_log_likelihood:.3f}; gap KL(q || posterior): "
        f"{posterior_kl:.3f}"
    )
    print(
        f"epsilon {0.3:.1f}; reparameterized h = 1 + sqrt(1) * epsilon: "
        f"{1 + np.sqrt(1) * 0.3:.3f}"
    )
    print(
        f"Jensen two-point Y=(1,4): log E[Y] {np.log(2.5):.3f}; E[log Y] "
        f"{(np.log(1) + np.log(4)) / 2:.3f}"
    )
# endregion book:fnd.generative-models.elbo-01


def linear_vae_terms(x, w, a, variance, noise_variance):
    """Exact means for q(h|x)=N(a.x, variance), p(x|h)=N(w*h, r*I)."""
    x, w, a = np.asarray(x), np.asarray(w), np.asarray(a)
    mu = x @ a
    residual = x - np.outer(mu, w)
    reconstruction = -0.5 * (
        x.shape[1] * np.log(2 * np.pi * noise_variance)
        + (np.mean(np.sum(residual**2, axis=1)) + np.dot(w, w) * variance) / noise_variance
    )
    kl = 0.5 * (np.mean(mu**2) + variance - 1 - np.log(variance))
    covariance = noise_variance * np.eye(x.shape[1]) + np.outer(w, w)
    log_likelihood = -0.5 * (
        x.shape[1] * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + np.mean(np.sum(x * np.linalg.solve(covariance, x.T).T, axis=1))
    )
    posterior_variance = noise_variance / (noise_variance + np.dot(w, w))
    posterior_mu = x @ w / (noise_variance + np.dot(w, w))
    posterior_kl = 0.5 * (
        (variance + np.mean((mu - posterior_mu) ** 2)) / posterior_variance
        - 1
        + np.log(posterior_variance / variance)
    )
    return {
        "reconstruction": float(reconstruction),
        "kl": float(kl),
        "elbo": float(reconstruction - kl),
        "log_likelihood": float(log_likelihood),
        "posterior_kl": float(posterior_kl),
    }


def linear_vae_gradient(x, w, a, variance, noise_variance):
    moment = x.T @ x / len(x)
    dw = (moment @ a - w * (a @ moment @ a + variance)) / noise_variance
    da = (moment @ w - (np.dot(w, w) + noise_variance) * moment @ a) / noise_variance
    dlog_variance = 0.5 * (1 - variance * (1 + np.dot(w, w) / noise_variance))
    return dw, da, dlog_variance


def train_linear_vae(seed=7, steps=2000, eta=0.01):
    rng = np.random.default_rng(seed)
    true_w, noise_variance = np.array([1.5, -0.8]), 0.5
    h = rng.normal(size=512)
    x = np.outer(h, true_w) + rng.normal(scale=np.sqrt(noise_variance), size=(len(h), 2))
    w, a, log_variance = np.array([0.1, 0.1]), np.zeros(2), 0.0
    curve = []
    for step in range(steps + 1):
        variance = float(np.exp(log_variance))
        if step % 10 == 0 or step == steps:
            curve.append({"step": step, **linear_vae_terms(x, w, a, variance, noise_variance)})
        if step < steps:
            dw, da, dv = linear_vae_gradient(x, w, a, variance, noise_variance)
            w, a, log_variance = w + eta * dw, a + eta * da, log_variance + eta * dv
    return {
        "seed": seed,
        "observations": len(x),
        "dimensions": 2,
        "true_w": true_w.tolist(),
        "noise_variance": noise_variance,
        "eta": eta,
        "training_steps": steps,
        "decoder_w": w.tolist(),
        "encoder_a": a.tolist(),
        "encoder_variance": float(np.exp(log_variance)),
        "steps": curve,
    }


# region book:fnd.generative-models.vae-01 output
if __name__ == "__main__":
    vae = train_linear_vae()
    print(
        f"linear Gaussian VAE: seed {vae['seed']}; {vae['observations']} points; {vae['dimensions']} dimensions"
    )
    print(
        f"data decoder w*: {vae['true_w']}; noise variance {vae['noise_variance']}; eta {vae['eta']}"
    )
    print("step: expected log-likelihood, KL to prior, bound, exact log-likelihood, gap")
    for row in vae["steps"]:
        if row["step"] in (0, 10, 50, 100, 500, 1000, 2000):
            print(
                f"{row['step']}: {row['reconstruction']:.6f}, {row['kl']:.6f}, {row['elbo']:.6f}, "
                f"{row['log_likelihood']:.6f}, {row['posterior_kl']:.6f}"
            )
# endregion book:fnd.generative-models.vae-01


def gan_optimum(p_data, p_g):
    """Best discriminator, game value and Jensen-Shannon divergence, in nats."""
    p_data, p_g = (np.asarray(p, dtype=float) for p in (p_data, p_g))
    if p_data.ndim != 1 or not p_data.size or p_data.shape != p_g.shape:
        raise ValueError("matching nonempty probability vectors required")
    for p in (p_data, p_g):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
            raise ValueError("finite nonnegative probabilities summing to one required")
    total = p_data + p_g
    # Off both supports, every discriminator is optimal; choose one half.
    discriminator = np.divide(
        p_data, total, out=np.full_like(total, 0.5), where=total > 0
    )
    value, divergence = 0.0, 0.0
    for p in (p_data, p_g):
        keep = p > 0  # The limit of p*log(p) at zero is zero.
        ratio = p[keep] / total[keep]
        value += float(p[keep] @ np.log(ratio))
        divergence += float(0.5 * (p[keep] @ np.log(2 * ratio)))
    return discriminator, value, divergence


# region book:fnd.generative-models.gan-01 output
from math import log

from demos.generative_models_demo import gan_optimum as optimum

if __name__ == "__main__":
    p_data = (0.5, 0.5)
    for p_g in ((0.8, 0.2), p_data):
        discriminator, value, divergence = optimum(p_data, p_g)
        print(f"data {p_data}; generator {p_g}")
        print(
            "best discriminator:", " ".join(f"{d:.6f}" for d in discriminator)
        )
        print(f"objective {value:.6f}; floor {-log(4):.6f}")
        print(f"Jensen-Shannon divergence {divergence:.6f}")
# endregion book:fnd.generative-models.gan-01


# region book:fnd.generative-models.diffusion-01 output
def noise_step(x, epsilon, beta):
    if not np.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("noise variance must be in [0, 1)")
    return np.sqrt(1 - beta) * np.asarray(x) + np.sqrt(beta) * np.asarray(
        epsilon
    )


def noise_jump(x, epsilon, betas):
    betas = np.asarray(betas, dtype=float)
    if (
        betas.ndim != 1
        or np.any((betas < 0) | (betas >= 1))
        or not np.isfinite(betas).all()
    ):
        raise ValueError("a vector of noise variances in [0, 1) is required")
    alpha_bar = np.prod(1 - betas)
    return np.sqrt(alpha_bar) * np.asarray(x) + np.sqrt(
        1 - alpha_bar
    ) * np.asarray(epsilon)


def noise_loss(epsilon, predicted):
    return float(np.sum((np.asarray(epsilon) - np.asarray(predicted)) ** 2))


if __name__ == "__main__":
    x, epsilon, beta = np.array([1.0, -2.0]), np.array([0.5, -1.0]), 0.36
    print(f"beta {beta}; x (1, -2); epsilon (0.5, -1)")
    print(
        "one step:", " ".join(f"{v:.3f}" for v in noise_step(x, epsilon, beta))
    )
    alpha_bar = (1 - beta) ** 3
    print(
        f"t=3 coefficients: {np.sqrt(alpha_bar):.3f} "
        f"{np.sqrt(1 - alpha_bar):.3f}"
    )
    print(
        "jump to t=3:",
        " ".join(f"{v:.3f}" for v in noise_jump(x, epsilon, [beta] * 3)),
    )
    print(
        f"predicted epsilon (0.4, -0.7); squared noise loss: "
        f"{noise_loss(epsilon, [0.4, -0.7]):.3f}"
    )
# endregion book:fnd.generative-models.diffusion-01


# region book:fnd.generative-models.reverse-01 output
from demos.output_wrap import print_wrapped


def reverse_step(x_t, predicted, betas, z):
    """Ho et al., algorithm 2: given epsilon prediction, sigma_t^2=beta_t."""
    betas = np.asarray(betas, dtype=float)
    if (
        betas.ndim != 1
        or not len(betas)
        or not np.isfinite(betas).all()
        or np.any((betas <= 0) | (betas >= 1))
    ):
        raise ValueError(
            "a nonempty vector of noise variances in (0, 1) is required"
        )
    x_t, predicted, z = (
        np.asarray(value, dtype=float) for value in (x_t, predicted, z)
    )
    if (
        x_t.shape != predicted.shape
        or x_t.shape != z.shape
        or not all(np.isfinite(value).all() for value in (x_t, predicted, z))
    ):
        raise ValueError(
            "state, prediction and noise need matching shapes and finite "
            "values"
        )
    beta, alpha_bar = betas[-1], np.prod(1 - betas)
    mean = (x_t - beta * predicted / np.sqrt(1 - alpha_bar)) / np.sqrt(
        1 - beta
    )
    return mean, mean + np.sqrt(beta) * z


if __name__ == "__main__":
    betas = np.full(3, 0.36)
    x_3 = noise_jump([1, -2], [0.5, -1], betas)
    predicted, z = np.array([0.4, -0.7]), np.array([0.25, -0.5])
    reverse_mean, x_2 = reverse_step(x_3, predicted, betas, z)
    print_wrapped(
        "t=3; beta 0.36; sigma 0.6; x_3:", " ".join(f"{v:.3f}" for v in x_3)
    )
    print_wrapped("given epsilon prediction (0.4, -0.7); fixed z (0.25, -0.5)")
    print_wrapped("reverse mean:", " ".join(f"{v:.3f}" for v in reverse_mean))
    print_wrapped("sampled x_2:", " ".join(f"{v:.3f}" for v in x_2))
    print_wrapped(
        "illustrative reverse step; no network was trained; exact jump used "
        "before rounding"
    )
# endregion book:fnd.generative-models.reverse-01


def noise_schedule(steps=1000):
    # Ho et al., section 4: https://arxiv.org/pdf/2006.11239
    betas = np.linspace(1e-4, 0.02, steps)
    alpha_bar = np.cumprod(1 - betas)
    return {
        "source": "Ho et al. (2020), section 4",
        "steps": steps,
        "beta_first": float(betas[0]),
        "beta_last": float(betas[-1]),
        "t": list(range(steps + 1)),
        "signal": np.r_[1, np.sqrt(alpha_bar)].tolist(),
        "noise": np.r_[0, np.sqrt(1 - alpha_bar)].tolist(),
    }


# region book:fnd.generative-models.schedule-01 output
if __name__ == "__main__":
    schedule = noise_schedule()
    print(
        f"linear schedule: {schedule['steps']} steps; beta {schedule['beta_first']} to {schedule['beta_last']}"
    )
    for t in (100, 250, 500, 750, 1000):
        print(f"t={t}: signal {schedule['signal'][t]:.3f}; noise {schedule['noise'][t]:.3f}")
# endregion book:fnd.generative-models.schedule-01


# region book:fnd.generative-models.scenarios-01 output
from demos.output_wrap import print_wrapped
from refund_agent.scenarios.suite import build_suite, list_suite

if __name__ == "__main__":
    as_of = datetime(2026, 9, 15, tzinfo=timezone.utc)
    listed = list_suite(as_of)
    generated = [build_suite(as_of, tau, 200, 7) for tau in (1, 2)]
    print_wrapped(
        "as of 2026-09-15; policy 2026-09-01; draw seed 7; training seed 7"
    )
    print_wrapped(
        "set: proposed rejected overruled distinct unseen-in-training kept"
    )
    for name, suite in [("listed", listed)] + [
        (f"temperature {tau}", s) for tau, s in zip((1, 2), generated)
    ]:
        counts = suite.manifest["counts"]
        fields = (
            "proposed", "rejected", "overruled", "distinct",
            "unseen_in_training", "kept",
        )
        values = " ".join(str(counts[key]) for key in fields)
        print_wrapped(f"{name}: {values}")
    print_wrapped(
        "distinct counts and unseen counts refer to the five case tokens, "
        "after schema rejection"
    )
    worked = next(
        row
        for row in generated[0].scenarios
        if row.tokens[:3] == ("changed_mind", True, "31-45")
        and row.label.outcome.value == "approve"
        and row.generator_decision == "deny"
    )
    print_wrapped(f"overruled case: {worked.tokens}")
    print_wrapped(
        f"request {worked.records.request.request_id}; amount "
        f"{worked.label.amount_cents} cents; tier {int(worked.label.tier)}"
    )
    delivery = next(
        e.at for e in worked.records.shipment.events if e.status == "delivered"
    )
    print_wrapped(
        f"delivered {delivery.date()}; days since delivery "
        f"{(as_of - delivery).days}; earlier refund {worked.tokens[4]}"
    )
    print_wrapped(
        f"generator {worked.generator_decision}; policy "
        f"{worked.label.outcome.value}; reason {worked.label.reason}; "
        f"{worked.label.policy_citation}"
    )
    print_wrapped(
        f"schema rejection example: {generated[0].rejected[0]['tokens']}; "
        f"{generated[0].rejected[0]['reason']}"
    )
    review = generated[0].review_sample(5, 31)
    print_wrapped(
        f"human review: pending; {len(review)} sampled rows, seed 31"
    )
# endregion book:fnd.generative-models.scenarios-01


AS_OF = datetime(2026, 9, 15, tzinfo=timezone.utc)


def exercise_fixture():
    """One planted defect: the entire report uses July's label policy in September."""
    import json

    from refund_agent.labels import relabel

    suite = build_suite(AS_OF, 1, 200, 7)
    data = training_data()
    cases = data.model_copy(
        update={
            "requests": [row.records.request for row in suite.scenarios],
            "orders": [row.records.order for row in suite.scenarios],
            "shipments": [row.records.shipment for row in suite.scenarios],
        }
    )
    stale = relabel(cases, "2026-07-01")
    rows = [
        {
            "draw": row.draw,
            "tokens": list(row.tokens),
            "request": row.records.request.model_dump(mode="json"),
            "order": row.records.order.model_dump(mode="json"),
            "shipment": row.records.shipment.model_dump(mode="json"),
            "generator_decision": row.generator_decision,
            "policy_label": label.model_dump(mode="json"),
            "policy_check_passed": row.generator_decision == label.outcome.value,
        }
        for row, label in zip(suite.scenarios, stale)
    ]
    manifest = json.loads(json.dumps(suite.manifest))
    manifest["policy_version"] = "2026-07-01"
    manifest["counts"]["overruled"] = sum(not row["policy_check_passed"] for row in rows)
    return {"manifest": manifest, "scenarios": rows, "rejected": list(suite.rejected)}


# region book:fnd.generative-models.keys-01 output
if __name__ == "__main__":
    print("exercise 1: positive temperature keeps network; greedy, top-2 and nucleus 0.8 do not")
    print(
        "exercise 2: exact autoregressive score; latent marginal integral bounded; GAN has no scoring density"
    )
    print("exercise 2: 1000 sequential DDPM passes; one VAE decoder pass")
    for variance in (1, 0.5):
        terms = linear_vae_terms(np.array([[2.0]]), np.array([1.0]), np.array([0.5]), variance, 1)
        print(
            f"exercise 3, q=N(1,{variance:g}): KL {terms['kl']:.3f}; E[log p(x|h)] {terms['reconstruction']:.3f}; "
            f"bound {terms['elbo']:.3f}; log p(x) {terms['log_likelihood']:.3f}; gap {terms['posterior_kl']:.3f}"
        )
    for name, rule in (
        ("tau 1", {"name": "temperature"}),
        ("tau 0.5", {"name": "temperature", "tau": 0.5}),
        ("top-2", {"name": "top-k", "k": 2}),
        ("nucleus 0.9", {"name": "nucleus", "p": 0.9}),
    ):
        p = sampling_distribution([3, 1, 0], rule)
        print(
            f"exercise 4a {name}: {' '.join(f'{v:.3f}' for v in p)}; {np.count_nonzero(p)} tokens"
        )
    print(
        "exercise 4b one step:", " ".join(f"{v:.3f}" for v in noise_step([2, 0], [-1, 0.5], 0.36))
    )
    print(
        "exercise 4b t=2:", " ".join(f"{v:.3f}" for v in noise_jump([2, 0], [-1, 0.5], [0.36] * 2))
    )
    print(f"exercise 4b loss: {noise_loss([-1, 0.5], [-0.8, 0.5]):.3f}")
    fixture = exercise_fixture()
    source = training_data()
    wrong = []
    for row in fixture["scenarios"]:
        label = policy_label(to_records(tuple(row["tokens"]), AS_OF, 7), source)
        if (row["policy_label"]["outcome"], row["policy_label"]["reason"]) != (
            label.outcome.value,
            label.reason,
        ):
            wrong.append(row)
    print(
        f"exercise 6: {fixture['manifest']['counts']['proposed']} draws; {len(fixture['scenarios'])} schema-valid rows; "
        f"{len(wrong)} stale labels; {sum(row['policy_check_passed'] for row in wrong)} wrong labels agree with the generator"
    )
    actual = [
        policy_label(to_records(tuple(row["tokens"]), AS_OF, 7), source).outcome.value
        for row in wrong
    ]
    print(
        f"exercise 6 corrections: deny to approve {actual.count('approve')}; deny to refer {actual.count('refer')}; "
        f"outdated policy citations {len(fixture['scenarios'])}"
    )
    print(
        "exercise 6 wrong draws and request ids:",
        " ".join(f"{row['draw']}:{row['request']['request_id']}" for row in wrong),
    )
    print(
        "exercise 7: add none plus five failures; failures refer with the failure as their reason"
    )
# endregion book:fnd.generative-models.keys-01


# region book:fnd.generative-models.keys-gan-01 output
from math import log

from demos.generative_models_demo import gan_optimum as optimum

if __name__ == "__main__":
    p_data, p_g = (0.6, 0.4), (0.3, 0.7)
    discriminator, value, divergence = optimum(p_data, p_g)
    print(f"exercise 5: data {p_data}; generator {p_g}")
    print("best discriminator:", " ".join(f"{d:.6f}" for d in discriminator))
    print(f"objective {value:.6f}; floor {-log(4):.6f}")
    print(f"Jensen-Shannon divergence {divergence:.6f}")
# endregion book:fnd.generative-models.keys-gan-01


def figure_data():
    scores = [2.0, 1.0, 0.5, 0.0, -1.0]
    panels = [
        ("temperature 0.5", {"name": "temperature", "tau": 0.5}),
        ("temperature 1", {"name": "temperature", "tau": 1}),
        ("temperature 2", {"name": "temperature", "tau": 2}),
        ("top-2", {"name": "top-k", "k": 2}),
        ("nucleus 0.8", {"name": "nucleus", "p": 0.8}),
    ]
    return {
        "sampling": {
            "tokens": ["deploy", "certificate", "DNS", "disk", "network"],
            "scores": scores,
            "true_cause": "certificate",
            "panels": [
                {
                    "name": name,
                    "rule": rule,
                    "probabilities": sampling_distribution(scores, rule).tolist(),
                }
                for name, rule in panels
            ],
        },
        "vae": train_linear_vae(),
        "schedule": noise_schedule(),
    }


def write_data(directory):
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "generative_models_figures.json": figure_data(),
        "generative_models_scenarios_06.json": exercise_fixture(),
    }
    for tau in (1, 2):
        build_suite(AS_OF, tau, 200, 7).write(
            directory / f"generative_models_scenarios_tau{tau}.json"
        )
    list_suite(AS_OF).write(directory / "generative_models_scenarios_listed.json")
    for name, value in artifacts.items():
        (directory / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-data", action="store_true")
    args = parser.parse_args()
    if args.write_data:
        write_data(Path(__file__).with_name("data"))
