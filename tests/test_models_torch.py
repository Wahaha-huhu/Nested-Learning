"""Torch tests (run on the pod: `pytest -q`). Skipped automatically where torch is missing."""
import pytest

torch = pytest.importorskip("torch")

from nlrepro.models import build_model, count_params                     # noqa: E402
from nlrepro.models.titans_selfmod import (selfmod_titans_reference,       # noqa: E402
                                           selfmod_titans_scan)

TINY_T = dict(arch="transformer", vocab_size=97, d_model=32, n_layers=2, n_heads=4,
              ffn_hidden=64, max_len=256)
TINY_H = dict(arch="hope", vocab_size=97, d_model=32, n_layers=2, n_heads=4,
              titans={"chunk": 8}, cms_levels=[{"kind": "incontext", "chunk": 16},
                                              {"kind": "static", "hidden": 48}])


def _inputs(B=2, H=3, L=37, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, H, L, d, generator=g, dtype=torch.float64)
    q = torch.nn.functional.normalize(torch.randn(B, H, L, d, generator=g, dtype=torch.float64), dim=-1)
    eta = 0.3 * torch.sigmoid(torch.randn(B, H, L, generator=g, dtype=torch.float64))
    la = torch.nn.functional.logsigmoid(3 + torch.randn(B, H, L, generator=g, dtype=torch.float64))
    M0 = torch.eye(d, dtype=torch.float64).expand(3, H, d, d) + 0.1 * torch.randn(
        3, H, d, d, generator=g, dtype=torch.float64)
    return u, q, eta, la, M0


@pytest.mark.parametrize("chunk,selfmod,selfval", [(5, True, False), (8, True, True),
                                                    (37, False, False), (1, True, False)])
def test_titans_scan_matches_token_loop(chunk, selfmod, selfval):
    args = _inputs()
    a = selfmod_titans_scan(*args, chunk, 0.1, 1, selfmod, selfval)
    b = selfmod_titans_reference(*args, chunk, 0.1, selfmod, selfval)
    assert torch.allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("cfg", [TINY_T, TINY_H])
def test_causal_and_trainable(cfg):
    torch.manual_seed(0)
    m = build_model(cfg).double()
    x = torch.randint(0, 97, (2, 40))
    y1 = m(x)
    x2 = x.clone()
    x2[:, 25:] = (x2[:, 25:] + 1) % 97
    y2 = m(x2)
    assert torch.allclose(y1[:, :25], y2[:, :25], atol=1e-10), "future tokens leak"
    loss = torch.nn.functional.cross_entropy(y1[:, :-1].reshape(-1, 97), x[:, 1:].reshape(-1))
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters without gradient: {dead}"


def test_hope_param_match_at_110m():
    import yaml
    from nlrepro.utils.common import load_config
    base = "experiments/e06_lm_from_scratch/configs/"
    t = count_params(build_model(load_config(base + "transformer_110m.yaml")["model"]))["total"]
    h = count_params(build_model(load_config(base + "hope_110m.yaml")["model"]))["total"]
    assert abs(h - t) / t < 0.02, (t, h)


def test_cms_adapter_schedule():
    transformers = pytest.importorskip("transformers")
    from nlrepro.models.hf_cms import ContinuumMemoryAdapter
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=6, num_attention_heads=4,
                                   num_key_value_heads=2)
    model = transformers.Qwen2ForCausalLM(cfg).float()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ad = ContinuumMemoryAdapter(model, [4, 8, 16], lr=1e-3)
    assert ad.level_of_layer == [0, 1, 2, 0, 1, 2]
    x = torch.randint(0, 64, (1, 12))
    for _ in range(16 // 4):
        model(input_ids=x, labels=x).loss.backward()
        ad.after_backward(4)
    assert ad.updates == [4, 2, 1]
    for n, p in model.named_parameters():
        changed = not torch.equal(before[n], p)
        assert changed == (".mlp." in n), n
