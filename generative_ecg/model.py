"""Train, reconstruct, and sample using the original ECG-SCD VAE modules."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .beats import SHAPE

LATENT = 512


def upstream(root):
    path = root / "02_Morphing" / "s05_utils_vae.py"
    spec = importlib.util.spec_from_file_location("ecg_scd_vae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def network(root):
    original, digest = upstream(root)
    encoder = original.CNNEncoder(LATENT)
    decoder = original.Decoder([100] * 4 + [int(np.prod(SHAPE))], use_bias=False)
    return original, digest, encoder, decoder


def train(root, data, out, epochs, batch_size, seed):
    from flax import serialization

    original, digest, encoder, decoder = network(root)
    out.mkdir(parents=True)
    train_x = np.load(data / "train.npy", mmap_mode="r")
    val_x = np.load(data / "val.npy", mmap_mode="r")
    key, enc_key, dec_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = (encoder.init(enc_key, jnp.zeros(SHAPE)), decoder.init(dec_key, jnp.zeros(LATENT)))
    steps = epochs * ((len(train_x) + batch_size - 1) // batch_size)
    schedule = optax.cosine_decay_schedule(1e-4, steps, alpha=0.05)
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)

    def objective(params, x, key):
        mu, variance = jax.vmap(encoder.apply, (None, 0))(params[0], x)
        z = original.gaussian_sample(key, mu, variance)
        decoded = jax.vmap(decoder.apply, (None, 0))(params[1], z).reshape(x.shape)
        rec = -original.gaussian_logpdf(decoded.reshape(len(x), -1), x.reshape(len(x), -1)).mean()
        kl = jax.vmap(original.gaussian_kl)(mu, variance).mean()
        return rec + 0.01 * kl, (rec, kl)

    @jax.jit
    def step(params, state, x, key):
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params, x, key)
        updates, state = optimizer.update(gradients, state)
        return optax.apply_updates(params, updates), state, metrics

    @jax.jit
    def validation(params, x):
        mu, _ = jax.vmap(encoder.apply, (None, 0))(params[0], x)
        prediction = jax.vmap(decoder.apply, (None, 0))(params[1], mu).reshape(x.shape)
        return jnp.mean((prediction - x)**2)

    best = float("inf")
    history = []
    rng = np.random.default_rng(seed)
    config = dict(upstream_sha256=digest, epochs=epochs, batch_size=batch_size, seed=seed,
                  latent=LATENT, shape=SHAPE, kl_weight=0.01)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for epoch in range(1, epochs + 1):
        totals = np.zeros(2)
        indices = rng.permutation(len(train_x))
        for start in range(0, len(indices), batch_size):
            x = train_x[indices[start:start + batch_size]]
            key, sample = jax.random.split(key)
            params, state, metrics = step(params, state, x, sample)
            totals += np.asarray(metrics) * len(x)
        val_sum = 0.0
        for start in range(0, len(val_x), batch_size):
            x = val_x[start:start + batch_size]
            val_sum += float(validation(params, x)) * len(x)
        mse = val_sum / len(val_x)
        if not np.isfinite(mse) or not np.isfinite(totals).all():
            raise RuntimeError("Non-finite training or validation metrics")
        if mse < best:
            best = mse
            temporary = out / "best.tmp"
            temporary.write_bytes(serialization.to_bytes(params))
            temporary.replace(out / "best.msgpack")
        rec, kl = totals / len(train_x)
        entry = dict(epoch=epoch, train_reconstruction=float(rec), train_kl=float(kl),
                     validation_mse=mse)
        history.append(entry)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(entry, flush=True)


def infer(root, data, model, out, count, seed):
    from flax import serialization

    original, digest, encoder, decoder = network(root)
    config = json.loads((model / "config.json").read_text())
    if config["upstream_sha256"] != digest:
        raise ValueError("Checkpoint requires the original training source revision")
    enc_key, dec_key = jax.random.split(jax.random.PRNGKey(seed))
    template = (encoder.init(enc_key, jnp.zeros(SHAPE)), decoder.init(dec_key, jnp.zeros(LATENT)))
    enc_params, dec_params = serialization.from_bytes(template, (model / "best.msgpack").read_bytes())
    out.mkdir(parents=True)
    encode = jax.jit(jax.vmap(encoder.apply, (None, 0)))
    decode = jax.jit(jax.vmap(decoder.apply, (None, 0)))
    test = np.load(data / "test.npy", mmap_mode="r")
    recon = []
    for start in range(0, len(test), 128):
        mu, _ = encode(enc_params, test[start:start + 128])
        recon.append(np.asarray(decode(dec_params, mu)).reshape(-1, *SHAPE))
    np.save(out / "reconstructed.npy", np.concatenate(recon))
    train_x = np.load(data / "train.npy", mmap_mode="r")
    means, variances = [], []
    for start in range(0, len(train_x), 128):
        mu, var = encode(enc_params, train_x[start:start + 128])
        means.append(np.asarray(mu))
        variances.append(np.asarray(var))
    mu, var = np.concatenate(means), np.concatenate(variances)
    km, kv, kz = jax.random.split(jax.random.PRNGKey(seed), 3)
    shape = (count, LATENT)
    sample_mean = mu.mean(0) + mu.std(0) * jax.random.normal(km, shape)
    sample_var = var.mean(0) + var.std(0) * jax.random.normal(kv, shape)
    z = original.gaussian_sample(kz, sample_mean, jnp.maximum(sample_var, 1e-6))
    synthetic = np.asarray(decode(dec_params, z)).reshape(-1, *SHAPE)
    if not np.isfinite(synthetic).all():
        raise RuntimeError("Non-finite generated beats")
    np.save(out / "synthetic.npy", synthetic)
    (out / "config.json").write_text(json.dumps(dict(count=count, seed=seed), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "infer"))
    parser.add_argument("upstream", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.count < 1:
        parser.error("Epochs, batch size, and sample count must be positive")
    if args.action == "train":
        train(args.upstream, args.data, args.model, args.epochs, args.batch_size, args.seed)
    else:
        if args.out is None:
            parser.error("--out is required for inference")
        infer(args.upstream, args.data, args.model, args.out, args.count, args.seed)
