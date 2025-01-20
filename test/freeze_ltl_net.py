import torch
import os

print(os.getcwd())
# state_dict1 = torch.load("experiments/ppo/FlatWorld-v0/tmp/1/ltl_net.pth")
# state_dict2 = torch.load("experiments/ppo/pretraining_FlatWorld-v0/seq/1/ltl_net.pth")

state_dict1 = torch.load("experiments/ppo/ChessWorld-v0/gcn/1/ltl_net.pth")
state_dict2 = torch.load("experiments/ppo/pretraining_ChessWorld-v0/gcn/1/ltl_net.pth")

if state_dict1.keys() != state_dict2.keys():
    print("The keys in the state dictionaries are different.")
else:
    print("The keys in the state dictionaries are the same.")


assert(state_dict1.keys() == state_dict2.keys())


all_equal = True

for key in state_dict1.keys():
    if not torch.equal(state_dict1[key], state_dict2[key]):
        print(f"Mismatch found at key: {key}")
        all_equal = False

if all_equal:
    print("The state dictionaries are equal.")
else:
    print("The state dictionaries are not equal.")

assert(all_equal)


all_close = True

for key in state_dict1.keys():
    if not torch.allclose(state_dict1[key], state_dict2[key], atol=1e-8):
        print(f"Mismatch (within tolerance) at key: {key}")
        all_close = False

if all_close:
    print("The state dictionaries are approximately equal (within tolerance).")
else:
    print("The state dictionaries are not equal, even within tolerance.")

assert(all_close)
