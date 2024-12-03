import torch
from SyntaxTree import SyntaxTree
import torch_geometric
import torch.nn.functional as F

var_names=[1]*5
assignment_vocab=None
VOCAB = [1]*15

sample_vars = ["PAD", "EPSILON", "NULL", "green", "red", "blue", "yellow", "orange", "aqua"]
sample_vocab = {0: "PAD", 1: "EPSILON", 2: "NULL", 3: "green", 4: "red", 5: "blue", 6: "blue&green", 7:"blue&red", 8: "blue&red&green", 9: "orange",
                10: "yellow", 11: "aqua", 12: "yellow&orange", 13: "aqua&red", 14: ""}

tree = SyntaxTree(sample_vars[3:], sample_vocab)
sample_reach = torch.tensor([[[5, 3, 6], [13, 0, 0], [6, 7, 8]], [[12, 13, 0], [5, 1, 0], [0, 0, 0]], [[8, 7, 0], [6, 7, 9], [0, 0, 0]]], dtype=torch.long)
sample_avoid = torch.tensor([[[9, 0, 0], [4, 11, 0], [10, 11, 0]], [[7, 6, 0], [11, 10, 0], [0, 0, 0]], [[10, 11, 0], [11, 0, 0], [0, 0, 0]]], dtype=torch.long)
lens = [3, 2, 2]

# print("Assignment mode")
# Xs, edges, rootss = tree.process_reach_avoid(sample_reach, lens, sample_avoid)
#
# print()
# print(Xs)
# print([[tree.embedding_dict[i] for i in x] for x in Xs])
# print(edges)
# print(rootss)
#
# print()
# print("Propositions mode")
Xs1, edges1, rootss1 = tree.process_reach_avoid(sample_reach, lens, sample_avoid,False)

# assignments = [(1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1), (0, 0, 0, 1)]

# TODO: add formula for avoiding sequences


# print(tree.minimal_formula(assignments))
# print(tree.syntax_tree(assignments))
print("Features")
print(Xs1)
print([tree.embedding_dict[i] for i in Xs1])

print()
print("Edges")
print(edges1)
print(rootss1)

# print(len(tree.embedding_dict) - len(tree.variable_names))

sample_embedding = torch.nn.Embedding(len(tree.embedding_dict), 16, padding_idx=0)
embedding_thing = sample_embedding(torch.tensor(Xs1, dtype=torch.long))

print()
print("Embedding")
print(embedding_thing)
print(embedding_thing.shape)


class GNN(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch_geometric.nn.GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

gnn = GNN(16, 2)
new_embeddings = gnn(embedding_thing, edges1)

print()
print("Through GNN")
print(new_embeddings)
print(new_embeddings.shape)

# good_roots = [[0, 24, 53], [6, 47], [60, 15]]
roots_embeddings = new_embeddings[rootss1]
# ordered_roots = new_embeddings[good_roots]

print()
print("Extract Root embeddings")
print(roots_embeddings)
# print(ordered_roots)
print(new_embeddings[rootss1].shape)






print()
print("Reshaped Tensor with lens")
max_len = max(lens)
dim_1 = len(rootss1) // max_len
seqs = new_embeddings[rootss1].view((dim_1, max_len, sample_embedding.embedding_dim))
print(seqs.shape)
print(seqs)

print()
print("Prepare for RNN")
seq = torch.nn.utils.rnn.pack_padded_sequence(seqs, torch.tensor(lens, dtype=torch.long), batch_first=True, enforce_sorted=False)
print(seq)

rnn = torch.nn.GRU(input_size=sample_embedding.embedding_dim, hidden_size=sample_embedding.embedding_dim, num_layers=1,
                          batch_first=True)

_, h = rnn(seq)
final_results = h[-1, ...]

print()
print("Final Results")
print(final_results)
print(final_results.shape)

