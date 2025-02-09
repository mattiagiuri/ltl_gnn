import torch
from torch import nn

from model.formulae_utils.SyntaxTreeStay import SyntaxTreeStay
from model.ltl.set_network import SetNetwork
from preprocessing import BatchedReachAvoidSequences, BatchedSequences, VOCAB, assignment_vocab, var_names
from model.ltl.gnn import GNN
from model.formulae_utils.SyntaxTree import SyntaxTree


class LTLNetGNN(nn.Module):
    def __init__(
            self,
            embedding: nn.Module,
            num_rnn_layers: int,
            num_gnn_layers: int,
            variable_names: list[str],
            assignment_vocabulary: dict,
            stay_mode: bool=False
    ):
        super().__init__()
        self.embedding = embedding
        embedding_dim = embedding.embedding_dim

        self.gnn = GNN(embedding_dim=embedding_dim, num_layers=num_gnn_layers)
        self.rnn = nn.GRU(input_size=2*embedding_dim, hidden_size=2*embedding_dim, num_layers=num_rnn_layers,
                          batch_first=True)

        self.embedding_dim = 2*embedding_dim

        if not stay_mode:
            self.syntax_treer = SyntaxTree(variable_names, assignment_vocabulary)
        else:
            self.syntax_treer = SyntaxTreeStay(variable_names, assignment_vocabulary)

    def forward(self, batched_seqs: tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]
                                    | BatchedReachAvoidSequences) -> torch.tensor:
        if isinstance(batched_seqs, BatchedReachAvoidSequences):
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs.all()
        else:
            (reach_lens, reach_data), (avoid_lens, avoid_data) = batched_seqs
        assert (reach_lens == avoid_lens).all()

        # X, edges, roots = self.syntax_treer.process_reach_avoid(reach_data, reach_lens, avoid_data)
        # X = torch.tensor(X, dtype=torch.long)
        #
        # embedded = self.embedding(X)
        # gnn_embedded = self.gnn(embedded, edges)

        # root_embeddings = gnn_embedded.view((dim_1, max_len, self.embedding_dim))
        # print("Reach")
        # print(reach_data)
        Xr, edgesr, rootsr = self.syntax_treer.process_sequence(reach_data, reach_lens)
        # print("Avoid")
        # print(avoid_data)
        Xa, edgesa, rootsa = self.syntax_treer.process_sequence(avoid_data, avoid_lens)

        # Xr = torch.tensor(Xr, dtype=torch.long)
        # Xa = torch.tensor(Xa, dtype=torch.long)

        embedded_reach = self.embedding(Xr)
        embedded_avoid = self.embedding(Xa)

        gnn_reach = self.gnn(embedded_reach, edgesr)
        gnn_avoid = self.gnn(embedded_avoid, edgesa)

        max_len = max(reach_lens)
        dim_1 = len(rootsr) // max_len

        root_embeddings_reach = gnn_reach[rootsr].view((dim_1, max_len, self.embedding.embedding_dim))
        root_embeddings_avoid = gnn_avoid[rootsa].view((dim_1, max_len, self.embedding.embedding_dim))

        root_embeddings = torch.cat((root_embeddings_reach, root_embeddings_avoid), dim=-1)

        seq = nn.utils.rnn.pack_padded_sequence(root_embeddings, reach_lens, batch_first=True, enforce_sorted=False)

        _, h = self.rnn(seq)
        return h[-1, ...]
