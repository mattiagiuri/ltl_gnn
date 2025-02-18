from src.ltl.automata import LDBA, ltl2ldba, ltl2nba
# from srcltl.automata.ldba import LDBA
# from visualize.automata import construct_ldba_chessworld
from ltl.logic import Assignment
from sequence.search import ExhaustiveSearch
from model.formulae_utils.ContextMaker import ContextMaker
import os


def construct_ldba_chessworld(formula: str, simplify_labels: bool = False, prune: bool = True, ldba: bool = True) -> LDBA:
    fun = ltl2ldba if ldba else ltl2nba
    ldba = fun(formula, simplify_labels=simplify_labels)
    print('Constructed LDBA.')
    # assert ldba.check_valid()
    # print('Checked valid.')
    if prune:
        # ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        props = {'queen', 'rook', 'bishop', 'knight', 'pawn'}
        ldba.prune([
            Assignment.where('queen', propositions=props),
            Assignment.where('rook', propositions=props),
            Assignment.where('knight', propositions=props),
            Assignment.where('bishop', propositions=props),
            Assignment.where('pawn', propositions=props),
            Assignment.where('queen', 'rook', propositions=props),
            Assignment.where('queen', 'bishop', propositions=props),
            Assignment.where('queen', 'pawn', 'bishop', propositions=props),
            Assignment.where('queen', 'rook', 'pawn', propositions=props),
            # Assignment.where('rook', 'pawn', propositions=props),
            Assignment.where('rook', 'knight', propositions=props),
            Assignment.where('rook', 'bishop', propositions=props),
            Assignment.where('knight', 'bishop', propositions=props),
            Assignment.zero_propositions(props),
        ])
        print('Pruned impossible transitions.')
    ldba.complete_sink_state()
    print('Added sink state.')
    ldba.compute_sccs()
    return ldba


def read_frozenset(fset):
    values = {piece: i for i, piece in zip(range(5), ['queen', 'pawn', 'knight', 'bishop', 'rook'])}
    assignments = list(sorted([name for name, status in fset if status], key=lambda s: values.get(s, float('inf'))))
    return "&".join(assignments)


def print_formulae_from_seqs(formula, inverse_vocab, context_maker, assignment_vocab):
    props = {'queen', 'rook', 'bishop', 'knight', 'pawn'}

    ldba = construct_ldba_chessworld(formula, simplify_labels=True, prune=True, ldba=True)

    search = ExhaustiveSearch(None, props, num_loops=1)
    seqs = search.all_sequences(ldba, ldba.initial_state)

    formula_seqs = []
    print()
    print(formula)

    for count, seq in enumerate(seqs):
        formula_seq = []
        c1 = 0
        for reach, avoid in seq:
            if isinstance(reach, int):
                reach_tup = (1,)
            else:
                reach_tup = tuple(sorted([inverse_vocab[read_frozenset(i)] for i in reach]))
            avoid_tup = tuple(sorted([inverse_vocab[read_frozenset(i)] for i in avoid]))

            try:
                reach_formula = context_maker.cache[reach_tup]
            except KeyError:
                print(f"reach {c1} of sequence {count} not in cache")
                # print(reach)
                print([assignment_vocab[i] for i in reach_tup])
                reach_formula = context_maker.make_formula(reach_tup)

            try:
                avoid_formula = context_maker.cache[avoid_tup]
            except KeyError:
                print(f"avoid {c1} of sequence {count} not in cache")
                # print(avoid)
                print([assignment_vocab[i] for i in avoid_tup])
                avoid_formula = context_maker.make_formula(avoid_tup)

            formula_seq.append((reach_formula, avoid_formula))
            c1 += 1

        formula_seqs.append(formula_seq)
        print(formula_seq)

    print()
    return formula_seqs


if __name__ == "__main__":
    sample_vocab = {0: 'PAD', 1: 'EPSILON', 2: 'NULL', 3: 'queen', 4: 'rook', 5: 'knight', 6: 'bishop', 7: 'pawn',
                        8: 'queen&rook', 9: 'queen&bishop', 10: 'queen&pawn&bishop', 11: 'queen&pawn&rook',
                        12: 'knight&rook', 13: 'bishop&rook', 14: 'knight&bishop', 15: 'blank'}
    inverse_vocab = {val: key for key, val in sample_vocab.items()}
    inverse_vocab[''] = 15

    var_names = ['EPSILON', 'NULL', 'queen', 'rook', 'knight', 'bishop', 'pawn', 'blank']
    true_vars = ['queen', 'rook', 'knight', 'bishop', 'pawn']

    context_maker = ContextMaker(sample_vocab, var_names, true_vars)
    context_maker.generate_cache()

    formulae = [
        "F (queen & (!knight U rook))",
        "F ((queen & bishop) & F ((knight & rook) & F bishop))",
        "(! knight U (queen & pawn & rook)) & (F(bishop & F (rook & knight)))",
        "(! (knight) U queen) & (!pawn U knight)",
        "F (G knight)",
        "(! (bishop | knight | rook | pawn) U queen)",
        "(!(knight | pawn) U queen) & (F(G rook))",
        "F (queen & F (knight & F (pawn)))",
        "(G F (queen & bishop)) & (G F (rook & knight))",
        "F (G (! queen))",
        "(queen => F rook) U (pawn | (rook & knight & (F bishop)))",
        "(G (F queen)) & (G ! pawn)"
    ]

    annoying_formulae = [
        "(!(knight) U queen) & (!pawn U knight)",
        "(!queen U pawn) & (! bishop U knight)",
        "(! rook U queen) & (! bishop U rook)",
        "(! pawn U knight) & (! rook U bishop)",
        "(! queen U bishop) & (! rook U pawn)"
    ]

    for formula in annoying_formulae:
        print_formulae_from_seqs(formula, inverse_vocab, context_maker, sample_vocab)
