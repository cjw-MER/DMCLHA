import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()

        inner_size = 4 * hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act("relu")

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.5)

    def get_hidden_act(self, act="relu"):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, heads))
        self.num_attention_heads = heads
        self.attention_head_size = int(hidden_size / heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key_1 = nn.Linear(hidden_size, self.all_head_size)
        self.value_1 = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(0.5)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12) # TODO
        self.out_dropout = nn.Dropout(0.5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):
        query_layer = self.query(input_tensor)
        key_layer_1 = self.key_1(input_tensor)
        value_layer_1 = self.value_1(input_tensor)

        query_layer = self.transpose_for_scores(query_layer).permute(0, 2, 1, 3)
        key_layer_1 = self.transpose_for_scores(key_layer_1).permute(0, 2, 3, 1)
        value_layer_1 = self.transpose_for_scores(value_layer_1).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer_1)
        attention_scores = attention_scores / self.sqrt_attention_head_size

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer_1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads):
        super(TransformerBlock, self).__init__()
        self.layer = MultiHeadAttention(hidden_dim, heads)
        self.feed_forward = FeedForward(hidden_dim)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, heads, layers):
        super(TransformerEncoder, self).__init__()
        block = TransformerBlock(hidden_dim, heads) # self attention
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])
    def forward(self, hidden_states_1, output_all_encoded_layers=False):
        for layer_module in self.blocks:
            hidden_states_1 = layer_module(hidden_states_1)
        return hidden_states_1