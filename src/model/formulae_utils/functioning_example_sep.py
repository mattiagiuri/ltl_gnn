import torch
from SyntaxTree import SyntaxTree
import torch_geometric
import torch.nn.functional as F
from visualize_trees import plot_graph

var_names=[1]*5
assignment_vocab=None
VOCAB = [1]*16

sample_vars = ["PAD", "EPSILON", "NULL", "green", "red", "blue", "yellow", "orange", "aqua", "magenta"]
# sample_vocab = {0: "PAD", 1: "EPSILON", 2: "NULL", 3: "green", 4: "red", 5: "blue", 6: "blue&green", 7:"blue&red", 8: "blue&red&green", 9: "orange",
#                 10: "yellow", 11: "aqua", 12: "yellow&orange", 13: "aqua&red", 14: ""}

sample_vocab = {0: "PAD", 1: "EPSILON", 2: "NULL", 3: "green", 4: "blue", 5: "aqua", 6: "blue&green", 7: "blue&aqua", 8: "blue&aqua&green", 9: "red&magenta",
                    10: "yellow", 11: "red", 12: "orange", 13: "magenta", 14: "green&aqua", 15: "blank"}

tree = SyntaxTree(sample_vars[3:], sample_vocab)

sample_reach = torch.tensor([[[5, 3, 14], [13, 0, 0], [6, 7, 8]], [[12, 13, 0], [5, 1, 0], [0, 0, 0]], [[8, 7, 0], [6, 7, 9], [0, 0, 0]]], dtype=torch.long)
sample_avoid = torch.tensor([[[0, 0, 0], [4, 11, 0], [10, 11, 0]], [[7, 6, 0], [11, 10, 0], [0, 0, 0]], [[10, 11, 0], [15, 0, 0], [0, 0, 0]]], dtype=torch.long)
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
print("Reach")
Xsr, edgesr, rootsr = tree.process_sequence(sample_reach, lens)
print("Avoid")
Xsa, edgesa, rootsa = tree.process_sequence(sample_avoid, lens)

# assignments = [(1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1), (0, 0, 0, 1)]

# TODO: add formula for avoiding sequences
# print(tree.embedding_dict)
# plot_graph(edgesr, {i: tree.embedding_dict[x] for i, x in enumerate(Xsr)})

# print(tree.minimal_formula(assignments))
# print(tree.syntax_tree(assignments))
print("Features reach")
print(Xsr)
print([tree.embedding_dict[i.item()] for i in Xsr])

print()
print("Features avoid")
print(Xsa)
print([tree.embedding_dict[i.item()] for i in Xsa])

print()
print("Edges and roots reach")
print(edgesr)
print(rootsr)

print()
print("Edges and roots avoid")
print(edgesa)
print(rootsa)

# print(len(tree.embedding_dict) - len(tree.variable_names))

sample_embedding = torch.nn.Embedding(len(tree.embedding_dict), 16, padding_idx=0)
embedding_reach = sample_embedding(Xsr)
embedding_avoid = sample_embedding(Xsa)

print()
print("Embedding")
# print(embedding_reach)
print(embedding_reach.shape)


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
new_embeddings_reach = gnn(embedding_reach, edgesr)
new_embeddings_avoid = gnn(embedding_avoid, edgesa)

print()
print("Through GNN")
# print(new_embeddings_reach)
print(new_embeddings_reach.shape)

# good_roots = [[0, 24, 53], [6, 47], [60, 15]]
roots_embeddings_reach = new_embeddings_reach[rootsr]
roots_embeddings_avoid = new_embeddings_avoid[rootsa]
# ordered_roots = new_embeddings[good_roots]

print()
print("Extract Root embeddings")
print(roots_embeddings_reach)
# print(ordered_roots)
print(roots_embeddings_reach.shape)






print()
print("Reshaped Tensor with lens")
max_len = max(lens)
dim_1 = len(rootsr) // max_len
seqs_reach = roots_embeddings_reach.view((dim_1, max_len, sample_embedding.embedding_dim))
seqs_avoid = roots_embeddings_avoid.view((dim_1, max_len, sample_embedding.embedding_dim))
print(seqs_reach)

print()
print("Concatanate representations")

seqs = torch.cat((seqs_reach, seqs_avoid), dim=-1)
print(seqs.shape)
print(seqs)

print()
print("Prepare for RNN")
seq = torch.nn.utils.rnn.pack_padded_sequence(seqs, torch.tensor(lens, dtype=torch.long), batch_first=True, enforce_sorted=False)
print(seq)

rnn = torch.nn.GRU(input_size=2*sample_embedding.embedding_dim, hidden_size=2*sample_embedding.embedding_dim, num_layers=1,
                          batch_first=True)

_, h = rnn(seq)
final_results = h[-1, ...]

print()
print("Final Results")
print(final_results)
print(final_results.shape)
