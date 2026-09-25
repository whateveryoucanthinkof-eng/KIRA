"""The all-rows memory read must stay bit-identical to the indexed gather it replaced."""
import torch


def _memory_with_history(n=1000, d=12, seed=0):
    torch.manual_seed(seed)
    gru = torch.nn.GRUCell(d, d)
    base = torch.nn.Parameter(torch.zeros(n, d), requires_grad=False)
    idx = torch.randperm(n)[: n // 3]
    # Memory rows written from a GRU output carry autograd history, as in training.
    base[idx] = gru(torch.randn(len(idx), d), base[idx])
    return gru, base


def test_forward_and_backward_match_indexed_gather():
    for seed in range(3):
        gru_a, mem_a = _memory_with_history(seed=seed)
        gru_b, mem_b = _memory_with_history(seed=seed)
        a = mem_a[list(range(mem_a.shape[0])), :]
        b = mem_b.clone()
        assert torch.equal(a, b)
        w = torch.randn_like(a)
        (a * w).sum().backward()
        (b * w).sum().backward()
        for pa, pb in zip(gru_a.parameters(), gru_b.parameters()):
            assert torch.equal(pa.grad, pb.grad)


def test_copy_is_independent_of_later_in_place_updates():
    _gru, mem = _memory_with_history()
    snap = mem.clone()
    before = snap.detach().clone()
    with torch.no_grad():
        mem[:5] = 7.0
    assert torch.equal(snap.detach(), before)
