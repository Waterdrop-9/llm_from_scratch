import torch
import pytest
import src.transformer as tf

def test_linear_layer():
    # 测试Linear层的前向传播
    in_features = 4
    out_features = 3
    batch_size = 2

    # 创建Linear层实例
    linear_layer = tf.Linear(in_features, out_features)
    W = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0]])

    expected_output = torch.tensor([[1.0, 5.0, 10.0],
                                    [5.0, 13.0, 26.0]], 
                                    dtype=torch.float32)

    # 创建一个固定输入张量，形状为[batch_size, in_features]
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    with torch.no_grad():
        linear_layer.weight.copy_(W)  # 将权重设置为固定值
    # 前向传播
    output_tensor = linear_layer(input_tensor)

    print(f"Input Tensor: {input_tensor}")
    print(f"Output Tensor: {output_tensor}")
    # 检查输出张量的形状是否正确
    assert output_tensor.shape == (batch_size, out_features), f"Expected output shape {(batch_size, out_features)}, but got {output_tensor.shape}"
    torch.testing.assert_close(output_tensor, expected_output)


def test_linear_preserves_leading_dimensions_and_registers_only_weight():
    linear_layer = tf.Linear(in_features=4, out_features=3)
    input_tensor = torch.randn(2, 5, 4)

    output_tensor = linear_layer(input_tensor)

    assert output_tensor.shape == (2, 5, 3)

    parameters = dict(linear_layer.named_parameters())
    assert set(parameters) == {"weight"}
    assert parameters["weight"].shape == (3, 4)


def test_embedding_looks_up_rows_and_preserves_token_shape():
    embedding = tf.Embedding(num_embeddings=4, embedding_dim=3)
    fixed_weight = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ]
    )
    token_ids = torch.tensor([[2, 0, 2], [1, 3, 0]], dtype=torch.long)
    expected_output = torch.tensor(
        [
            [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0], [6.0, 7.0, 8.0]],
            [[3.0, 4.0, 5.0], [9.0, 10.0, 11.0], [0.0, 1.0, 2.0]],
        ]
    )

    with torch.no_grad():
        embedding.weight.copy_(fixed_weight)

    output = embedding(token_ids)

    assert output.shape == (2, 3, 3)
    torch.testing.assert_close(output, expected_output)

    parameters = dict(embedding.named_parameters())
    assert set(parameters) == {"weight"}
    assert parameters["weight"].shape == (4, 3)


def test_embedding_accumulates_gradients_for_repeated_token_ids():
    embedding = tf.Embedding(num_embeddings=4, embedding_dim=3)
    token_ids = torch.tensor([[0, 2, 2]], dtype=torch.long)

    embedding(token_ids).sum().backward()

    expected_gradient = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert embedding.weight.grad is not None
    torch.testing.assert_close(embedding.weight.grad, expected_gradient)


def test_rmsnorm_matches_a_hand_calculated_example():
    norm = tf.RMSNorm(d_model=4, eps=0.0)
    input_tensor = torch.tensor([[[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
    expected_output = torch.tensor([[[1.2, 3.2, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]])

    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    output = norm(input_tensor)

    assert output.shape == input_tensor.shape
    torch.testing.assert_close(output, expected_output)

    parameters = dict(norm.named_parameters())
    assert set(parameters) == {"weight"}
    assert parameters["weight"].shape == (4,)


def test_rmsnorm_normalizes_scale_along_the_last_dimension():
    norm = tf.RMSNorm(d_model=4, eps=0.0)
    input_tensor = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 4.0, -3.0]]]
    )

    output = norm(input_tensor)
    scaled_output = norm(input_tensor * 10.0)
    output_rms = torch.sqrt(output.pow(2).mean(dim=-1))

    torch.testing.assert_close(output, scaled_output)
    torch.testing.assert_close(output_rms, torch.ones_like(output_rms))


def test_rmsnorm_preserves_input_dtype_and_backpropagates():
    norm = tf.RMSNorm(d_model=4)
    low_precision_input = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float16
    )

    low_precision_output = norm(low_precision_input)

    assert low_precision_output.dtype == torch.float16

    input_tensor = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32, requires_grad=True
    )
    norm(input_tensor).square().sum().backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_silu_matches_pytorch_and_backpropagates():
    input_tensor = torch.tensor(
        [[-10.0, -1.0, 0.0, 1.0, 10.0]], requires_grad=True
    )

    output = tf.silu(input_tensor)
    expected_output = torch.nn.functional.silu(input_tensor)

    assert output.shape == input_tensor.shape
    assert output.dtype == input_tensor.dtype
    assert output.device == input_tensor.device
    torch.testing.assert_close(output, expected_output)

    output.sum().backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_swiglu_preserves_shape_and_registers_three_projections():
    swiglu = tf.SwiGLU(d_model=4, d_ff=6)
    input_tensor = torch.randn(2, 3, 4)

    output = swiglu(input_tensor)

    assert output.shape == input_tensor.shape

    parameters = dict(swiglu.named_parameters())
    assert set(parameters) == {"w1.weight", "w2.weight", "w3.weight"}
    assert parameters["w1.weight"].shape == (6, 4)
    assert parameters["w2.weight"].shape == (6, 4)
    assert parameters["w3.weight"].shape == (4, 6)


def test_swiglu_matches_a_hand_calculated_identity_projection():
    swiglu = tf.SwiGLU(d_model=2, d_ff=2)
    identity = torch.eye(2)
    input_tensor = torch.tensor([[[-1.0, 0.0], [1.0, 2.0]]])
    expected_output = torch.tensor(
        [[[0.26894143, 0.0], [0.7310586, 3.5231884]]]
    )

    with torch.no_grad():
        swiglu.w1.weight.copy_(identity)
        swiglu.w2.weight.copy_(identity)
        swiglu.w3.weight.copy_(identity)

    output = swiglu(input_tensor)

    torch.testing.assert_close(output, expected_output)


def test_swiglu_backpropagates_through_all_three_projections():
    swiglu = tf.SwiGLU(d_model=4, d_ff=6)
    input_tensor = torch.randn(2, 3, 4, requires_grad=True)

    swiglu(input_tensor).square().mean().backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for parameter in swiglu.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_embedding_rmsnorm_swiglu_composition_backpropagates():
    embedding = tf.Embedding(num_embeddings=8, embedding_dim=4)
    norm = tf.RMSNorm(d_model=4)
    swiglu = tf.SwiGLU(d_model=4, d_ff=6)
    token_ids = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=torch.long)

    output = swiglu(norm(embedding(token_ids)))
    output.square().mean().backward()

    assert output.shape == (2, 3, 4)
    for module in (embedding, norm, swiglu):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_softmax_matches_pytorch_along_the_requested_dimension():
    input_tensor = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]],
            [[2.0, 0.0, 1.0], [3.0, -2.0, 1.0]],
        ]
    )

    for dim in (0, 1, -1):
        output = tf.softmax(input_tensor, dim=dim)

        torch.testing.assert_close(output, torch.softmax(input_tensor, dim=dim))
        torch.testing.assert_close(
            output.sum(dim=dim), torch.ones_like(output.sum(dim=dim))
        )


def test_softmax_is_stable_for_large_inputs():
    input_tensor = torch.tensor([[10_000.0, 10_001.0, 10_002.0]])

    output = tf.softmax(input_tensor, dim=-1)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.softmax(input_tensor, dim=-1))


def test_scaled_dot_product_attention_matches_a_masked_example():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[10.0, 1.0], [2.0, 20.0]]])
    mask = torch.tensor([[True, False], [True, True]])

    output = tf.scaled_dot_product_attention(q, k, v, mask)

    scale = 2**-0.5
    second_row_weights = torch.softmax(torch.tensor([0.0, scale]), dim=-1)
    expected = torch.stack(
        (
            v[0, 0],
            second_row_weights[0] * v[0, 0]
            + second_row_weights[1] * v[0, 1],
        )
    ).unsqueeze(0)
    torch.testing.assert_close(output, expected)


def test_scaled_dot_product_attention_supports_4d_broadcasting_and_backward():
    q = torch.randn(2, 3, 2, 4, requires_grad=True)
    k = torch.randn(2, 3, 5, 4, requires_grad=True)
    v = torch.randn(2, 3, 5, 6, requires_grad=True)
    mask = torch.tensor(
        [[True, False, True, False, False], [True, True, True, False, True]]
    )

    output = tf.scaled_dot_product_attention(q, k, v, mask)
    expanded_mask_output = tf.scaled_dot_product_attention(
        q, k, v, mask.expand(2, 3, 2, 5)
    )

    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    expected_probabilities = torch.softmax(
        expected_scores.masked_fill(~mask, float("-inf")), dim=-1
    )
    expected = torch.matmul(expected_probabilities, v)

    assert output.shape == (2, 3, 2, 6)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output, expanded_mask_output)

    output.square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_rope_registers_nonpersistent_caches():
    rope = tf.RoPE(theta=10_000.0, d_k=4, max_seq_len=6)

    buffers = dict(rope.named_buffers())

    assert set(buffers) == {"cos_cache", "sin_cache"}
    assert buffers["cos_cache"].shape == (6, 2)
    assert buffers["sin_cache"].shape == (6, 2)
    assert dict(rope.named_parameters()) == {}
    assert "cos_cache" not in rope.state_dict()
    assert "sin_cache" not in rope.state_dict()


def test_rope_position_zero_is_identity():
    rope = tf.RoPE(theta=10_000.0, d_k=4, max_seq_len=4)
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    token_positions = torch.tensor([[0]])

    output = rope(input_tensor, token_positions)

    torch.testing.assert_close(output, input_tensor)


def test_rope_matches_a_hand_calculated_rotation():
    rope = tf.RoPE(theta=1.0, d_k=4, max_seq_len=2)
    input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    token_positions = torch.tensor([[1]])
    cosine = torch.cos(torch.tensor(1.0))
    sine = torch.sin(torch.tensor(1.0))
    expected = torch.tensor(
        [
            [
                [
                    cosine - 2 * sine,
                    sine + 2 * cosine,
                    3 * cosine - 4 * sine,
                    3 * sine + 4 * cosine,
                ]
            ]
        ]
    )

    output = rope(input_tensor, token_positions)

    torch.testing.assert_close(output, expected)


def test_rope_supports_batched_token_positions_and_backward():
    rope = tf.RoPE(theta=10_000.0, d_k=4, max_seq_len=6)
    input_tensor = torch.randn(2, 3, 4, requires_grad=True)
    token_positions = torch.tensor([[0, 1, 2], [3, 4, 5]])

    output = rope(input_tensor, token_positions)

    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_rope_rejects_an_odd_embedding_dimension():
    with pytest.raises(ValueError, match="must be even"):
        tf.RoPE(theta=10_000.0, d_k=3, max_seq_len=4)


def test_multihead_self_attention_matches_two_head_example():
    attention = tf.MultiHeadSelfAttention(d_model=4, num_heads=2)
    identity = torch.eye(4)
    input_tensor = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]]
    )

    with torch.no_grad():
        attention.q_proj.weight.copy_(identity)
        attention.k_proj.weight.copy_(identity)
        attention.v_proj.weight.copy_(identity)
        attention.output_proj.weight.copy_(identity)

    first_weight, second_weight = torch.softmax(
        torch.tensor([0.0, 2**-0.5]), dim=-1
    )
    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [first_weight, second_weight, second_weight, first_weight],
            ]
        ]
    )

    output = attention(input_tensor)

    torch.testing.assert_close(output, expected)


def test_multihead_self_attention_is_causal():
    torch.manual_seed(0)
    attention = tf.MultiHeadSelfAttention(d_model=8, num_heads=2)
    original = torch.randn(1, 5, 8)
    changed_future = original.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:]) * 100

    original_output = attention(original)
    changed_output = attention(changed_future)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_multihead_self_attention_rope_default_positions_match_explicit_positions():
    attention = tf.MultiHeadSelfAttention(
        d_model=8,
        num_heads=2,
        max_seq_len=5,
        theta=10_000.0,
    )
    input_tensor = torch.randn(2, 5, 8)
    token_positions = torch.arange(5).expand(2, 5)

    default_output = attention(input_tensor)
    explicit_output = attention(input_tensor, token_positions=token_positions)

    assert default_output.shape == input_tensor.shape
    torch.testing.assert_close(default_output, explicit_output)


def test_multihead_self_attention_backpropagates_through_all_projections():
    attention = tf.MultiHeadSelfAttention(
        d_model=8,
        num_heads=2,
        max_seq_len=4,
        theta=10_000.0,
    )
    input_tensor = torch.randn(2, 4, 8, requires_grad=True)

    attention(input_tensor).square().mean().backward()

    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.output_proj,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


def test_multihead_self_attention_rejects_indivisible_head_dimension():
    with pytest.raises(ValueError, match="must be divisible"):
        tf.MultiHeadSelfAttention(d_model=7, num_heads=2)


@pytest.mark.parametrize(
    ("max_seq_len", "theta"),
    [(4, None), (None, 10_000.0)],
)
def test_multihead_self_attention_rejects_incomplete_rope_configuration(
    max_seq_len: int | None,
    theta: float | None,
):
    with pytest.raises(ValueError, match="Both theta and max_seq_len"):
        tf.MultiHeadSelfAttention(
            d_model=8,
            num_heads=2,
            max_seq_len=max_seq_len,
            theta=theta,
        )


def test_transformer_block_residual_path_preserves_input_when_branches_are_zero():
    block = tf.TransformerBlock(
        d_model=8,
        num_heads=2,
        d_ff=12,
        max_seq_len=5,
        theta=10_000.0,
    )
    input_tensor = torch.randn(2, 5, 8)

    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w3.weight.zero_()

    output = block(input_tensor)

    torch.testing.assert_close(output, input_tensor)


def test_transformer_block_is_causal():
    torch.manual_seed(1)
    block = tf.TransformerBlock(
        d_model=8,
        num_heads=2,
        d_ff=12,
        max_seq_len=5,
        theta=10_000.0,
    )
    original = torch.randn(1, 5, 8)
    changed_future = original.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:]) * 100

    original_output = block(original)
    changed_output = block(changed_future)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_transformer_block_backpropagates_through_all_parameters():
    block = tf.TransformerBlock(
        d_model=8,
        num_heads=2,
        d_ff=12,
        max_seq_len=4,
        theta=10_000.0,
    )
    input_tensor = torch.randn(2, 4, 8, requires_grad=True)
    token_positions = torch.arange(4).expand(2, 4)

    output = block(input_tensor, token_positions=token_positions)
    output.square().mean().backward()

    assert output.shape == input_tensor.shape
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_transformer_lm_maps_token_ids_to_logits_and_registers_all_layers():
    model = tf.TransformerLM(
        vocab_size=17,
        context_length=6,
        d_model=8,
        num_layers=3,
        num_heads=2,
        d_ff=12,
        rope_theta=10_000.0,
    )
    token_ids = torch.randint(0, 17, (2, 5))

    logits = model(token_ids)

    assert logits.shape == (2, 5, 17)
    assert isinstance(model.layers, torch.nn.ModuleList)
    assert len(model.layers) == 3
    parameter_roots = {name.split(".")[0] for name, _ in model.named_parameters()}
    assert parameter_roots == {"token_embeddings", "layers", "ln_final", "lm_head"}


def test_transformer_lm_enforces_context_length_boundary():
    model = tf.TransformerLM(
        vocab_size=17,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=12,
        rope_theta=10_000.0,
    )

    assert model(torch.randint(0, 17, (1, 4))).shape == (1, 4, 17)
    with pytest.raises(ValueError, match="exceeds model context length"):
        model(torch.randint(0, 17, (1, 5)))


def test_transformer_lm_prefix_logits_do_not_depend_on_future_tokens():
    torch.manual_seed(2)
    model = tf.TransformerLM(
        vocab_size=17,
        context_length=5,
        d_model=8,
        num_layers=2,
        num_heads=2,
        d_ff=12,
        rope_theta=10_000.0,
    )
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 10, 11]])

    original_logits = model(original)
    changed_logits = model(changed_future)

    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])


def test_transformer_lm_backpropagates_through_all_parameters():
    model = tf.TransformerLM(
        vocab_size=17,
        context_length=4,
        d_model=8,
        num_layers=2,
        num_heads=2,
        d_ff=12,
        rope_theta=10_000.0,
    )
    token_ids = torch.randint(0, 17, (2, 4))

    model(token_ids).square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_causal_mask_fault_injection_detects_future_leakage(monkeypatch):
    torch.manual_seed(3)
    model = tf.TransformerLM(
        vocab_size=17,
        context_length=5,
        d_model=8,
        num_layers=2,
        num_heads=2,
        d_ff=12,
        rope_theta=10_000.0,
    )
    model.eval()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 10, 11]])
    prefix_length = 3

    with torch.inference_mode():
        original_logits = model(original)
        changed_logits = model(changed_future)
    baseline_diff = (
        original_logits[:, :prefix_length] - changed_logits[:, :prefix_length]
    ).abs().max().item()

    original_sdpa = tf.scaled_dot_product_attention

    def attention_without_mask(q, k, v, mask=None):
        return original_sdpa(q, k, v, mask=None)

    monkeypatch.setattr(tf, "scaled_dot_product_attention", attention_without_mask)

    with torch.inference_mode():
        leaky_original_logits = model(original)
        leaky_changed_logits = model(changed_future)
    no_mask_diff = (
        leaky_original_logits[:, :prefix_length]
        - leaky_changed_logits[:, :prefix_length]
    ).abs().max().item()

    print(f"baseline prefix max diff: {baseline_diff:.8f}")
    print(f"no-mask prefix max diff: {no_mask_diff:.8f}")
    assert baseline_diff <= 1e-6
    assert no_mask_diff > 1e-5
