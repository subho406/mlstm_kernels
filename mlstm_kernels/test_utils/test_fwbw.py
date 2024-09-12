from typing import Optional
import torch


def test_forward(
    f1,
    f2,
    inputs: tuple[torch.Tensor],
    comp_func=torch.allclose,
    comp_func_kwargs={},
    show_diff_func=None,
):
    out1 = f1(*inputs)
    out2 = f2(*inputs)

    if out1.shape != out2.shape:
        print("Bad output shape")

    if not comp_func(out1, out2, **comp_func_kwargs):
        print("Difference")
        if show_diff_func is not None:
            show_diff_func(out1, out2)


def test_backward(
    f1,
    f2,
    inputs: tuple[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    comp_func=torch.allclose,
    comp_func_kwargs={},
    show_diff_func=None,
):
    inputs1 = [inp.clone().detach() for inp in inputs]
    inputs2 = [inp.clone().detach() for inp in inputs]
    for inp in inputs1:
        inp.requires_grad_(True)
    for inp in inputs2:
        inp.requires_grad_(True)

    out1 = f1(*inputs1)
    out2 = f2(*inputs2)

    if mask is None:
        mask = torch.randn_like(out1)
    mask1 = mask.clone().detach()
    mask2 = mask.clone().detach()

    l1 = (out1 * mask1).sum()
    l1.backward()
    l2 = (out2 * mask2).sum()
    l2.backward()

    for n, (inp1, inp2) in enumerate(zip(inputs1, inputs2)):
        if not comp_func(inp1.grad, inp2.grad, **comp_func_kwargs):
            print(f"Difference in {n}-th gradient")
            if show_diff_func is not None:
                show_diff_func(inp1.grad, inp2.grad)
