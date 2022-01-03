import einops
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange


class Attention(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512, **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = tf.keras.layers.Dense(inner_dim, input_dim=dim, use_bias=False)
        self.to_kv = tf.keras.layers.Dense(inner_dim * 2, input_dim=dim, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim, input_dim=inner_dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = tf.keras.layers.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, context=None, mask=None, context_mask=None):
        n = inputs.shape[-2]
        heads = self.heads
        max_pos_emb = self.max_pos_emb
        if context is None:
            has_context = False
            context = inputs
        else:
            has_context = True

        q, k, v = tf.split((self.to_q(inputs), *self.to_kv(context)), 2, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        dots = tf.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        seq = tf.range(n)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = (
            tf.clip_by_value(
                dist, clip_value_min=-max_pos_emb, clip_value_max=max_pos_emb
            )
            + max_pos_emb
        )
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = tf.einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if mask is not None or context_mask is not None:
            if mask is not None:
                mask = tf.ones(*inputs.shape[:2])
            if not has_context:
                if context_mask is None:
                    context_mask = mask
            else:
                if context_mask is None:
                    context_mask = tf.ones(*context.shape[:2])
            mask_value = -tf.experimental.numpy.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots = tf.where(mask, mask_value, dots)

        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)
