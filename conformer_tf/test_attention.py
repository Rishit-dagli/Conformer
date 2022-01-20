import random

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .attention import Attention


class AttentionTest(tf.test.TestCase):
    def setUp(self):
        super(AttentionTest, self).setUp()

        self.attention = Attention(
            dim=512, heads=8, dim_head=64, dropout=0.1, max_pos_emb=512
        )

    def generate_param_list():
        param_list = []
        for n in range(10):
            param = random.randint(1, 20)
            param_list.append([param, param])
        return param_list

    @parameterized.expand(generate_param_list())
    def test_shape_and_rank(self, n_input, n_output):
        input_shape = [n_input, 1024, 512]
        output_shape = [n_output, 1024, 512]
        input = tf.random.uniform(input_shape)
        output = self.attention(input)

        self.assertEqual(tf.rank(output), 3)
        self.assertShapeEqual(np.zeros(output_shape), output)


if __name__ == "__main__":
    tf.test.main()
