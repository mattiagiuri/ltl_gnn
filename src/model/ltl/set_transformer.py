import torch
import torch.nn as nn


class SetTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super(SetTransformer, self).__init__()

        # self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)
        seqs_length = x.size(1)
        set_sizes = x.size(2)
        final_dim = x.size(3)

        # print("x shape", x.shape)

        # cls_tokens = self.cls_token.expand(batch_size, seqs_length, -1, -1)

        # x_with_cls = torch.cat([cls_tokens, x], dim=2)
        x_reshaped = x.reshape(batch_size * seqs_length, set_sizes, final_dim)
        x_reshaped = self.ln2(x_reshaped)

        attn_output, _ = self.mhsa(x_reshaped, x_reshaped, x_reshaped)
        attn_output = x_reshaped + attn_output
        attn_output = self.ln1(attn_output)

        ff_output = self.ff(attn_output)
        ff_output = attn_output + ff_output
        # ff_output = self.ln2(ff_output)

        # cls_out = ff_output[:, 0, :]
        #
        # cls_out = cls_out.reshape(batch_size, seqs_length, final_dim)

        global_out = ff_output.mean(dim=-2).reshape(batch_size, seqs_length, final_dim)

        return global_out


if __name__ == "__main__":
    test_model = SetTransformer(16)

    print(test_model.cls_token)

    batch = 5
    x = torch.tensor([[[[1 for a in range(16)] for b in range(4)] for j in range(1, 6)] for k in range(5)], dtype=torch.float)
    print(x.shape)

    out = test_model(x)

    print(out.shape)
    print(out)