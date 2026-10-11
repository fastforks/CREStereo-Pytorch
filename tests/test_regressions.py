import torch
from nets.update import BasicUpdateBlock


def test_prepared_context_tracks_checkpoint_reload():
    torch.manual_seed(17)
    block = BasicUpdateBlock(hidden_dim=8, cor_planes=36).eval().prepare_inference()
    inp = torch.randn(1, 8, 4, 5)
    h = torch.randn_like(inp)
    motion = torch.randn(1, 128, 4, 5)
    state = {name: value.clone() for name, value in block.state_dict().items()}
    state["gru.convz1.weight"].add_(0.01)
    block.load_state_dict(state)
    with torch.no_grad():
        actual = block._gru_fused(h, block.context(inp), motion)
        expected = block.gru(h, torch.cat((inp, motion), 1))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
