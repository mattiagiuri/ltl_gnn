import argparse
import time
import datetime
from torch import nn

import preprocessing
import torch_ac
import sys

import utils
from ltl import EventuallySampler
from model.ltl.ltl_embedding import LtlEmbedding
from model.model import Model
from model.policy.continuous_actor import ContinuousActor
from utils import torch_utils
from envs import make_env
from utils.logging.file_logger import FileLogger
from train.experiment_metadata import ExperimentMetadata
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore


def main():
    # Parse arguments

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--ltl-sampler", default="Default",
                        help="the ltl formula template to sample from (default: DefaultSampler)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{SAMPLER}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="number of updates between two logs (default: 10)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=1,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=2 * 10 ** 8,
                        help="number of frames of training (default: 2*10e8)")
    parser.add_argument("--checkpoint-dir", default=None)

    ## Evaluation parameters
    parser.add_argument("--eval", action="store_true", default=False,
                        help="evaluate the saved model (default: False)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="number of episodes to evaluate on (default: 5)")
    parser.add_argument("--eval-env", default=None,
                        help="name of the environment to train on (default: use the same \"env\" as training)")
    parser.add_argument("--ltl-samplers-eval", default=None, nargs='+',
                        help="the ltl formula templates to sample from for evaluation (default: use the same \"ltl-sampler\" as training)")
    parser.add_argument("--eval-procs", type=int, default=1,
                        help="number of processes (default: use the same \"procs\" as training)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ignoreLTL", action="store_true", default=False,
                        help="the network ignores the LTL input")
    parser.add_argument("--noLTL", action="store_true", default=False,
                        help="the environment no longer has an LTL goal. --ignoreLTL must be specified concurrently.")
    parser.add_argument("--progression-mode", default="full",
                        help="Full: uses LTL progression; partial: shows the propositions which progress or falsify the formula; none: only original formula is seen. ")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--gnn", default="RGCN_8x32_ROOT_SHARED",
                        help="use gnn to model the LTL (only if ignoreLTL==True)")
    parser.add_argument("--int-reward", type=float, default=0.0,
                        help="the intrinsic reward for LTL progression (default: 0.0)")
    parser.add_argument("--pretrained-gnn", action="store_true", default=False, help="load a pre-trained LTL module.")
    parser.add_argument("--dumb-ac", action="store_true", default=False, help="Use a single-layer actor-critic")
    parser.add_argument("--freeze-ltl", action="store_true", default=False,
                        help="Freeze the gradient updates of the LTL module")

    args = parser.parse_args()

    use_mem = False

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    model_name = args.model
    storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir

    experiment = ExperimentMetadata(
        algorithm=args.algo,
        env=args.env,
        name=model_name,
        seed=args.seed,
        config={
            'test': 'test'
        }
    )

    txt_logger = TextLogger(experiment)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.set_seed(args.seed)

    # Set device

    device = "cpu"
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        assert args.ltl_sampler == 'eventually_sampler'
        envs.append(make_env(args.env, EventuallySampler))

    # Sync environments
    envs[0].reset(seed=args.seed)

    txt_logger.info("Environments loaded\n")

    # Load training status
    model_store = ModelStore(experiment)
    resuming = False
    try:
        status = model_store.load_training_status()
        txt_logger.info("Resuming training from existing run.\n")
        resuming = True
    except FileNotFoundError:
        status = {"num_frames": 0, "update": 0}

    def build_model():
        obs_dim = envs[0].observation_space['features'].shape[0]
        action_dim = envs[0].action_space.shape[0]
        env_embedding_dim = 64
        env_net = torch_utils.make_mlp_layers([obs_dim, 128, env_embedding_dim], activation=nn.Tanh)
        ltl_net = LtlEmbedding(5, 32)
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[64 + 32, 64, 64, 64],
                                activation=dict(
                                    hidden=nn.ReLU,
                                    output=nn.Tanh
                                ),
                                state_dependent_std=True)
        critic = torch_utils.make_mlp_layers([64 + 32, 64, 64, 1], activation=nn.Tanh, final_layer_activation=False)
        return Model(env_net, ltl_net, actor, critic)

    model = build_model()
    if "model_state" in status:
        model.load_state_dict(status["model_state"])
        txt_logger.info("Loaded model from existing run.\n")

    model.to(device)
    # txt_logger.info("{}\n".format(model))

    # Load algo
    algo = torch_ac.PPO(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocessing.preprocess_obss)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Loaded optimizer from existing run.\n")

    # init the evaluator
    if args.eval:
        raise NotImplementedError('TODO: implement eval')
        # eval_samplers = args.ltl_samplers_eval if args.ltl_samplers_eval else [args.ltl_sampler]
        # eval_env = args.eval_env if args.eval_env else args.env
        # eval_procs = args.eval_procs if args.eval_procs else args.procs
        #
        # evals = []
        # for eval_sampler in eval_samplers:
        #     evals.append(utils.Eval(eval_env, model_name, eval_sampler,
        #                             seed=args.seed, device=device, num_procs=eval_procs, ignoreLTL=args.ignoreLTL,
        #                             progression_mode=progression_mode, gnn=args.gnn, dumb_ac=args.dumb_ac))

    # Train model

    print(f'Num parameters: {sum([p.numel() for p in model.parameters()])}')
    # sys.exit(0)

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    file_logger = FileLogger(experiment, resuming=resuming)
    loggers = [txt_logger, file_logger]
    use_wandb = False
    if use_wandb:
        wandb_logger = WandbLogger(experiment, project_name='deep-ltl', resuming=resuming)
        loggers.append(wandb_logger)
    logger = MultiLogger(*loggers)

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            remaining_duration = int((args.frames - num_frames) / fps)
            remaining_time = str(datetime.timedelta(seconds=remaining_duration))

            average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"],
                                                                    logs["num_frames_per_episode"])
            average_discounted_return = utils.average_discounted_return(logs["return_per_episode"],
                                                                        logs["num_frames_per_episode"], args.discount)
            logs.update({
                "arps": average_reward_per_step,
                "adr": average_discounted_return,
                'frames': num_frames,
                'fps': fps,
                'remaining': remaining_time,
            })
            del logs['num_frames']
            logger.log(logs)

            status["num_frames"] = num_frames

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": algo.model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            # utils.save_status(status, model_dir + "/train")
            model_store.save_training_status(status)
            txt_logger.info("Status saved")

            if args.eval:
                raise NotImplementedError()
                # # we send the num_frames to align the eval curves with the training curves on TB
                # for evalu in evals:
                #     evalu.eval(num_frames, episodes=args.eval_episodes)


def __main__():
    main()
