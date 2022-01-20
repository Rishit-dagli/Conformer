import random

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .conformer_tf import ConformerBlock, ConformerConvModule


class ConformerBlockTest(tf.test.TestCase):
    def setUp(self):
        super(ConformerBlockTest, self).setUp()

        self.model = ConformerBlock(
            dim=512,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.0,
            ff_dropout=0.0,
            conv_dropout=0.0,
        )

    def generate_param_list():
        param_list = []
        for a in range(3):
            param_list.append([a, a])
        return param_list

    @parameterized.expand(generate_param_list())
    def test_shape_and_rank(self, n_input, n_output):
        input_shape = [n_input, 1024, 512]
        output_shape = [n_output, 1024, 512]
        input = tf.random.uniform(input_shape)
        output = self.model(input)

        self.assertEqual(tf.rank(output), 3)
        self.assertShapeEqual(np.zeros(output_shape), output)


class ConformerConvModuleTest(tf.test.TestCase):
    def setUp(self):
        super(ConformerConvModuleTest, self).setUp()

        self.layer = ConformerConvModule(
            dim=512,
            causal=False,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.1,
        )

    def generate_param_list():
        param_list = []
        for a in range(3):
            param_list.append([a, a])
        return param_list

    @parameterized.expand(generate_param_list())
    def test_shape_and_rank(self, n_input, n_output):
        input_shape = [n_input, 1024, 512]
        output_shape = [1, 1024, 512]
        input = tf.random.uniform(input_shape)
        output = self.layer(input)

        self.assertEqual(tf.rank(output), 3)
        self.assertShapeEqual(np.zeros(output_shape), output)


if __name__ == "__main__":
    tf.test.main()
