# Solver for Dynamic VRPTW, baseline strategy is to use the static solver HGS-VRPTW repeatedly
import argparse
import os
import sys
import time

import numpy as np

import tools
from baselines.strategies import STRATEGIES
from environment import ControllerEnvironment, VRPEnvironment
from tools import get_config, log, solve_static_vrptw

saved_state = {}


def run_oracle(args, env):
    # Oracle strategy which looks ahead, this is NOT a feasible strategy but gives a 'bound' on the performance
    # Bound written with quotes because the solution is not optimal so a better solution may exist
    # This oracle can also be used as supervision for training a model to select which requests to dispatch

    # First get hindsight problem (each request will have a release time)
    # As a start solution for the oracle solver, we use the greedy solution
    # This may help the oracle solver to find a good solution more quickly
    log("Running greedy baseline to get start solution and hindsight problem for oracle solver...")
    run_baseline(args, env, strategy='greedy')
    # Get greedy solution as simple list of routes
    greedy_solution = [route for epoch, routes in env.final_solutions.items() for route in routes]
    hindsight_problem = env.get_hindsight_problem()

    # Compute oracle solution (separate time limit since epoch_tlim is used for greedy initial solution)
    log(f"Start computing oracle solution with {len(hindsight_problem['coords'])} requests...")
    oracle_solution = min(solve_static_vrptw(hindsight_problem, time_limit=args.oracle_tlim, initial_solution=greedy_solution), key=lambda x: x[1])[0]
    oracle_cost = tools.validate_static_solution(hindsight_problem, oracle_solution)
    log(f"Found oracle solution with cost {oracle_cost}")

    # Run oracle solution through environment (note: will reset environment again with same seed)
    total_reward = run_baseline(args, env, oracle_solution=oracle_solution)
    assert -total_reward == oracle_cost, "Oracle solution does not match cost according to environment"
    return total_reward


def run_baseline(args, env, oracle_solution=None, strategy=None):
    strategy = strategy or args.strategy

    rng = np.random.default_rng(args.solver_seed)

    total_reward = 0
    done = False
    # Note: info contains additional info that can be used by your solver
    observation, static_info = env.reset()
    start_time = time.time()
    epoch_tlim = static_info['epoch_tlim']
    if static_info["is_static"]:
        size = len(observation["epoch_instance"]['demands'])
        if size <= 300:
            problem_type = 'SA'
        elif size <= 500:
            problem_type = 'SB'
        else:
            problem_type = 'SC'
    else:
        n = static_info['end_epoch'] - static_info['start_epoch'] + 1  # 5 ~ 8
        problem_type = f'D{n}'
    log(f'Size: {len(observation["epoch_instance"]["demands"])} , Dynamic: {not static_info["is_static"]} , Time limit: {epoch_tlim}, Type: {problem_type}')
    cfgs = [
        [
            (15, 'C'),  # C 13ins 274284.54
            {'seed': 1, 'nbG': 70, 'mTCV': 10, 'nbC': 10, 'mTTW': 180, 'iTWP': 1.0, 'tF': 0.15, 'rP': 60, 'fGNFS': 0.25, 'mCSSD': 15, 'iPLS': 17, 'nbE': 8, 'minP': 35, 'incP': 40}
        ],
        [
            (10, 'B'),  # B 40ins 186432.13
            {'seed': 1, 'nbG': 35, 'mTCV': 30, 'nbC': 5, 'mTTW': 60, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.1, 'mCSSD': 12, 'iPLS': 11, 'nbE': 5, 'minP': 45, 'incP': 35}
        ],
        [
            (8, 'C'),  # C 13ins 274122.62
            {'seed': 43, 'nbG': 30, 'mTCV': 25, 'nbC': 5, 'mTTW': 480, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.05, 'mCSSD': 16, 'iPLS': 10, 'nbE': 3, 'minP': 25, 'incP': 50}
        ],
        [
            (5, 'A'),  # A 40ins 129964.05
            {'seed': 1, 'nbG': 35, 'mTCV': 30, 'nbC': 6, 'mTTW': 180, 'iTWP': 1.0, 'tF': 0.2, 'rP': 70, 'fGNFS': 0.2, 'mCSSD': 16, 'iPLS': 12, 'nbE': 6, 'minP': 45, 'incP': 20}
        ],
        [
            (5, 'B'),  # B 40ins 186537.775
            {'seed': 1, 'nbG': 35, 'mTCV': 15, 'nbC': 1, 'mTTW': 540, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.05, 'mCSSD': 20, 'iPLS': 11, 'nbE': 5, 'minP': 50, 'incP': 35}
        ],
        [
            (3, 'A'),  # A 40ins 129937.3
            {'seed': 1, 'nbG': 35, 'mTCV': 50, 'nbC': 3, 'mTTW': 180, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.05, 'mCSSD': 11, 'iPLS': 10, 'nbE': 6, 'minP': 45, 'incP': 30}
        ],
        [
            (2, 'A'),  # A 40ins 130043.38
            {'seed': 1, 'nbG': 30, 'mTCV': 15, 'nbC': 9, 'mTTW': 360, 'iTWP': 1.0, 'tF': 0.3, 'rP': 30, 'fGNFS': 0.1, 'mCSSD': 19, 'iPLS': 20, 'nbE': 5, 'minP': 50, 'incP': 20}
        ],
        [
            (1, 'A'),  # A 40ins 130170.275
            {'seed': 1, 'nbG': 35, 'mTCV': 30, 'nbC': 5, 'mTTW': 420, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.05, 'mCSSD': 12, 'iPLS': 11, 'nbE': 5, 'minP': 35, 'incP': 20}
        ]
    ]
    if args.default:
        config_str = f' -seed {args.solver_seed}'
    elif args.standard:
        config_str = get_config(
            **{'seed': 1, 'nbG': 35, 'mTCV': 30, 'nbC': 5, 'mTTW': 420, 'iTWP': 1.0, 'tF': 0.2, 'rP': 50, 'fGNFS': 0.05, 'mCSSD': 12, 'iPLS': 11, 'nbE': 5, 'minP': 35, 'incP': 20}
        )
    else:
        cfg = min(cfgs, key=lambda x: (abs(epoch_tlim - x[0][0] * 60), abs(ord(problem_type[1]) - ord(x[0][1]))))
        log(f'Using config for {cfg[0][0]}-min {cfg[0][1]}-type')
        config_str = get_config(
            **cfg[1]
        )
    if problem_type == 'SB' and args.simple_init_smart:
        args.simple_init = True
        log('Using simple init')
    if problem_type[0] == 'D':
        args.num_keep_2 = 1000
        args.num_iter = 10000
    num_requests_postponed = 0
    while not done:
        epoch_instance = observation['epoch_instance']

        if args.verbose:
            log('=' * 20 + f"\nEpoch {static_info['start_epoch']} <= {observation['current_epoch']} <= {static_info['end_epoch']}", newline=False)
            num_requests_open = len(epoch_instance['request_idx']) - 1
            num_new_requests = num_requests_open - num_requests_postponed
            log(f" | Requests: +{num_new_requests:3d} = {num_requests_open:3d}, {epoch_instance['must_dispatch'].sum():3d}/{num_requests_open:3d} must-go...", newline=False, flush=True)

        if oracle_solution is not None:
            request_idx = set(epoch_instance['request_idx'])
            epoch_solution = [route for route in oracle_solution if len(request_idx.intersection(route)) == len(route)]
            cost = tools.validate_dynamic_epoch_solution(epoch_instance, epoch_solution)
        else:
            # Select the requests to dispatch using the strategy
            global saved_state
            saved_state['initial_solution'] = None
            epoch_instance_dispatch = STRATEGIES[strategy](
                ins=epoch_instance, rng=rng, args=vars(args), static=static_info['dynamic_context'], saved_state=saved_state,
                start_epoch=static_info['start_epoch'], end_epoch=static_info['end_epoch'], current_epoch=observation['current_epoch'],
                current_time=observation['current_time'], planning_starttime=observation['planning_starttime'],
                min_insert=args.min_insert, max_left=args.max_left, max_time_increase=args.max_time_increase
            )
            n_total = epoch_instance["request_idx"].size - 1
            n_dispatch = max(0, epoch_instance_dispatch["request_idx"].size - 1)
            log(f' Pre-solving time: {time.time() - start_time:.3f}s, {n_dispatch}/{n_total} dispatched and {n_total-n_dispatch}/{n_total} postponed')

            # Run HGS with time limit and get last solution (= best solution found)
            # Note we use the same solver_seed in each epoch: this is sufficient as for the static problem
            # we will exactly use the solver_seed whereas in the dynamic problem randomness is in the instance
            epoch_solution, cost = min(solve_static_vrptw(
                epoch_instance_dispatch, time_limit=epoch_tlim - round(time.time() - start_time), it=args.num_iter, num_keep_1=args.num_keep_1, num_keep_2=args.num_keep_2,
                debug=args.debug, config_str=config_str, initial_solution=saved_state['initial_solution'], simple_init=args.simple_init
            ), key=lambda x: x[-1])

            # Map HGS solution to indices of corresponding requests
            epoch_solution = [epoch_instance_dispatch['request_idx'][route] for route in epoch_solution]
            t = time.time() - start_time
            log(f'Epoch time: {t:.3f}s, wastes {epoch_tlim-t:.3f}s, cost={cost}')

        # Submit solution to environment
        observation, reward, done, info = env.step(epoch_solution)
        start_time = time.time()
        assert cost is None or reward == -cost, ("Reward should be negative cost of solution", reward, cost)
        assert not info['error'], f"Environment error: {info['error']}"

        total_reward += reward

    if args.verbose:
        log(f"Cost of solution: {-total_reward}")

    return total_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=['greedy', 'random', 'lazy', 'ppo'], default='ppo', help="Baseline strategy used to decide whether to dispatch routes")
    # Note: these arguments are only for convenience during development, during testing you should use controller.py
    parser.add_argument("--instance", help="Instance to solve")
    parser.add_argument("--instance_seed", type=int, default=1, help="Seed to use for the dynamic instance")
    parser.add_argument("--solver_seed", type=int, default=1, help="Seed to use for the solver")
    parser.add_argument("--static", action='store_true', help="Add this flag to solve the static variant of the problem (by default dynamic)")
    parser.add_argument("--epoch_tlim", type=int, default=-1, help="Time limit per epoch")
    parser.add_argument("--oracle_tlim", type=int, default=120, help="Time limit for oracle")
    parser.add_argument("--verbose", action='store_true', help="Show verbose output")
    parser.add_argument("--debug", action='store_true', help="显示更多调试信息")
    parser.add_argument("--random_p", type=float, default=0.5)
    parser.add_argument("--default", action='store_true', help="对比默认行为")
    parser.add_argument("--standard", action='store_true', help="使用标准参数")
    parser.add_argument("--simple_init", action='store_true', help="为HGS提供简单初始解")
    parser.add_argument("--simple_init_smart", action='store_true', help="为HGS提供简单初始解")
    parser.add_argument("--num_iter", type=int, default=100000, help="HGS多少轮未提升则重启")
    parser.add_argument("--num_keep_1", type=int, default=0, help="HGS重启时保留几个最优合法解")
    parser.add_argument("--num_keep_2", type=int, default=0, help="HGS重启时保留几个最优非法解")
    parser.add_argument('--cuda', type=str, default='')
    args = parser.parse_args()

    args.f_weight = [float(i) for i in args.f_weight.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.instance is not None:
        env = VRPEnvironment(seed=args.instance_seed, instance=tools.read_vrplib(args.instance), epoch_tlim=args.epoch_tlim, is_static=args.static)
    else:
        assert args.strategy != "oracle", "Oracle can not run with external controller"
        # Run within external controller
        env = ControllerEnvironment(sys.stdin, sys.stdout)

    # Make sure these parameters are not used by your solver
    args.instance = None
    args.instance_seed = None
    args.static = None
    args.epoch_tlim = None

    run_baseline(args, env)

    if args.instance is not None:
        log(tools.json_dumps_np(env.final_solutions))
