import json
import os
import platform
import subprocess
import sys
from io import StringIO
import time

import numpy as np

IS_LOCAL = os.path.exists('IS_LOCAL')
if not IS_LOCAL:
    sys.stderr.write('Logging is off\n')
    sys.stderr.flush()

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable


def log(*obj, newline=True, flush=True):
    # Write logs to stderr since program uses stdout to communicate with controller
    if not IS_LOCAL:
        return
    print(*obj, end='\n' if newline else '', file=sys.stderr, flush=flush)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_dumps_np(data):
    return json.dumps(data, cls=NumpyJSONEncoder)


def json_loads_np(json_string):
    return lists_to_np(json.loads(json_string))


def lists_to_np(obj):
    """Function will convert lists to numpy recursively."""
    if isinstance(obj, dict):
        return {k: lists_to_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return np.array(obj)
    return obj


def to_giant_tour(routes, with_depot=True):
    # Converts a list of routes to a giant tour with depots
    if len(routes) == 0:
        return []
    tour = list(routes[0])
    for route in routes[1:]:
        if with_depot:
            tour.append(0)
        tour.extend(list(route))
    return tour


def compute_solution_driving_time(instance, solution):
    return sum([
        compute_route_driving_time(route, instance['duration_matrix'])
        for route in solution
    ])


def validate_static_solution(instance, solution, allow_skipped_customers=False):

    if not allow_skipped_customers:
        validate_all_customers_visited(solution, len(instance['coords']) - 1)

    for route in solution:
        validate_route_capacity(route, instance['demands'], instance['capacity'])
        validate_route_time_windows(route, instance['duration_matrix'], instance['time_windows'], instance['service_times'])

    return compute_solution_driving_time(instance, solution)


def validate_dynamic_epoch_solution(epoch_instance, epoch_solution):
    """
    Validates a solution for a VRPTW instance, raises assertion if not valid
    Returns total driving time (excluding waiting time)
    """

    # Renumber requests (and depot) to 0,1...n
    request_idx = epoch_instance['request_idx']
    assert request_idx[0] == 0
    assert (request_idx[1:] > request_idx[:-1]).all()
    # Look up positions of request idx
    solution = [np.searchsorted(request_idx, route) for route in epoch_solution]

    # Check that all 'must_dispatch' requests are dispatched
    # if 'must_dispatch' in instance:
    must_dispatch = epoch_instance['must_dispatch'].copy()
    for route in solution:
        must_dispatch[route] = False
    assert not must_dispatch.any(), f"Some requests must be dispatched but were not: {request_idx[must_dispatch]}"

    static_instance = {
        k: v for k, v in epoch_instance.items()
        if k not in ('request_idx', 'customer_idx', 'must_dispatch')
    }

    return validate_static_solution(static_instance, solution, allow_skipped_customers=True)


def compute_route_driving_time(route, duration_matrix):
    """Computes the route driving time excluding waiting time between stops"""
    # From depot to first stop + first to last stop + last stop to depot
    return duration_matrix[0, route[0]] + duration_matrix[route[:-1], route[1:]].sum() + duration_matrix[route[-1], 0]


def validate_all_customers_visited(solution, num_customers):
    flat_solution = np.array([stop for route in solution for stop in route])
    assert len(flat_solution) == num_customers, "Not all customers are visited"
    visited = np.zeros(num_customers + 1)  # Add padding for depot
    visited[flat_solution] = True
    assert visited[1:].all(), "Not all customers are visited"


def validate_route_capacity(route, demands, capacity):
    assert sum(demands[route]) <= capacity, f"Capacity validated for route, {sum(demands[route])} > {capacity}"


def validate_route_time_windows(route, dist, timew, service_t, release_t=None):
    depot = 0  # For readability, define variable
    earliest_start_depot, latest_arrival_depot = timew[depot]
    if release_t is not None:
        earliest_start_depot = max(earliest_start_depot, release_t[route].max())
    current_time = earliest_start_depot + service_t[depot]

    prev_stop = depot
    for stop in route:
        earliest_arrival, latest_arrival = timew[stop]
        arrival_time = current_time + dist[prev_stop, stop]
        # Wait if we arrive before earliest_arrival
        current_time = max(arrival_time, earliest_arrival)
        assert current_time <= latest_arrival, f"Time window violated for stop {stop}: {current_time} not in ({earliest_arrival}, {latest_arrival})"
        current_time += service_t[stop]
        prev_stop = stop
    current_time += dist[prev_stop, depot]
    assert current_time <= latest_arrival_depot, f"Time window violated for depot: {current_time} not in ({earliest_start_depot}, {latest_arrival_depot})"


def readlines(filename):
    try:
        with open(filename, 'r') as f:
            return f.readlines()
    except:
        with open(filename, 'rb') as f:
            return [line.decode('utf-8', errors='ignore').strip() for line in f.readlines()]


def read_vrptw_solution(filename, return_extra=False):
    """Reads a VRPTW solution in VRPLib format (one route per row)"""
    solution = []
    extra = {}

    for line in readlines(filename):
        if line.startswith('Route'):
            solution.append(np.array([int(node) for node in line.split(":")[-1].strip().split(" ")]))
        else:
            if len(line.strip().split(" ")) == 2:
                key, val = line.strip().split(" ")
                extra[key] = val

    if return_extra:
        return solution, extra
    return solution


def read_vrplib(filename, rounded=True):
    loc = []
    demand = []
    mode = ''
    capacity = None
    edge_weight_type = None
    edge_weight_format = None
    duration_matrix = []
    service_t = []
    timewi = []
    with open(filename, 'r') as f:

        for line in f:
            line = line.strip(' \t\n')
            if line == "":
                continue
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(" : ")[1])
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(" : ")[1]
            elif line.startswith('EDGE_WEIGHT_FORMAT'):
                edge_weight_format = line.split(" : ")[1]
            elif line == 'NODE_COORD_SECTION':
                mode = 'coord'
            elif line == 'DEMAND_SECTION':
                mode = 'demand'
            elif line == 'DEPOT_SECTION':
                mode = 'depot'
            elif line == "EDGE_WEIGHT_SECTION":
                mode = 'edge_weights'
                assert edge_weight_type == "EXPLICIT"
                assert edge_weight_format == "FULL_MATRIX"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "EOF":
                break
            elif mode == 'coord':
                node, x, y = line.split()  # Split by whitespace or \t, skip duplicate whitespace
                node = int(node)
                x, y = (int(x), int(y)) if rounded else (float(x), float(y))

                if node == 1:
                    depot = (x, y)
                else:
                    assert node == len(loc) + 2  # 1 is depot, 2 is 0th location
                    loc.append((x, y))
            elif mode == 'demand':
                node, d = [int(v) for v in line.split()]
                if node == 1:
                    assert d == 0
                demand.append(d)
            elif mode == 'edge_weights':
                duration_matrix.append(list(map(int if rounded else float, line.split())))
            elif mode == 'service_t':
                node, t = line.split()
                node = int(node)
                t = int(t) if rounded else float(t)
                if node == 1:
                    assert t == 0
                assert node == len(service_t) + 1
                service_t.append(t)
            elif mode == 'time_windows':
                node, l, u = line.split()
                node = int(node)
                l, u = (int(l), int(u)) if rounded else (float(l), float(u))
                assert node == len(timewi) + 1
                timewi.append([l, u])

    return {
        'is_depot': np.array([1] + [0] * len(loc), dtype=bool),
        'coords': np.array([depot] + loc),
        'demands': np.array(demand),
        'capacity': capacity,
        'time_windows': np.array(timewi),
        'service_times': np.array(service_t),
        'duration_matrix': np.array(duration_matrix) if len(duration_matrix) > 0 else None
    }


def write_vrplib(filename, instance, name="problem", euclidean=False, is_vrptw=True):
    # LKH/VRP does not take floats (HGS seems to do)

    coords = instance['coords']
    demands = instance['demands']
    is_depot = instance['is_depot']
    duration_matrix = instance['duration_matrix']
    capacity = instance['capacity']
    assert (np.diag(duration_matrix) == 0).all()
    assert (demands[~is_depot] > 0).all()

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in [
                ("NAME", name),
                ("COMMENT", "ORTEC"),  # For HGS we need an extra row...
                ("TYPE", "CVRP"),
                ("DIMENSION", len(coords)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D" if euclidean else "EXPLICIT"),
            ] + ([] if euclidean else [
                ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
            ]) + [("CAPACITY", capacity)]
        ]))
        f.write("\n")

        if not euclidean:
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(map(str, row)))
                f.write("\n")

        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate(coords)
        ]))
        f.write("\n")

        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demands)
        ]))
        f.write("\n")

        f.write("DEPOT_SECTION\n")
        for i in np.flatnonzero(is_depot):
            f.write(f"{i+1}\n")
        f.write("-1\n")

        if is_vrptw:

            service_t = instance['service_times']
            timewi = instance['time_windows']

            # Following LKH convention
            f.write("SERVICE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, s)
                for i, s in enumerate(service_t)
            ]))
            f.write("\n")

            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, l, u)
                for i, (l, u) in enumerate(timewi)
            ]))
            f.write("\n")

            if 'release_times' in instance:
                release_times = instance['release_times']

                f.write("RELEASE_TIME_SECTION\n")
                f.write("\n".join([
                    "{}\t{}".format(i + 1, s)
                    for i, s in enumerate(release_times)
                ]))
                f.write("\n")

        f.write("EOF\n")


def str_vrplib(instance, name="problem", euclidean=False, is_vrptw=True):
    # LKH/VRP does not take floats (HGS seems to do)

    coords = instance['coords']
    demands = instance['demands']
    is_depot = instance['is_depot']
    duration_matrix = instance['duration_matrix']
    capacity = instance['capacity']
    assert (np.diag(duration_matrix) == 0).all()
    assert (demands[~is_depot] > 0).all()

    f = StringIO()
    f.write("\n".join([
        "{} : {}".format(k, v)
        for k, v in [
            ("NAME", name),
            ("COMMENT", "ORTEC"),  # For HGS we need an extra row...
            ("TYPE", "CVRP"),
            ("DIMENSION", len(coords)),
            ("EDGE_WEIGHT_TYPE", "EUC_2D" if euclidean else "EXPLICIT"),
        ] + ([] if euclidean else [
            ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
        ]) + [("CAPACITY", capacity)]
    ]))
    f.write("\n")

    if not euclidean:
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in duration_matrix:
            f.write("\t".join(map(str, row)))
            f.write("\n")

    f.write("NODE_COORD_SECTION\n")
    f.write("\n".join([
        "{}\t{}\t{}".format(i + 1, x, y)
        for i, (x, y) in enumerate(coords)
    ]))
    f.write("\n")

    f.write("DEMAND_SECTION\n")
    f.write("\n".join([
        "{}\t{}".format(i + 1, d)
        for i, d in enumerate(demands)
    ]))
    f.write("\n")

    f.write("DEPOT_SECTION\n")
    for i in np.flatnonzero(is_depot):
        f.write(f"{i+1}\n")
    f.write("-1\n")

    if is_vrptw:

        service_t = instance['service_times']
        timewi = instance['time_windows']

        # Following LKH convention
        f.write("SERVICE_TIME_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, s)
            for i, s in enumerate(service_t)
        ]))
        f.write("\n")

        f.write("TIME_WINDOW_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, l, u)
            for i, (l, u) in enumerate(timewi)
        ]))
        f.write("\n")

        if 'release_times' in instance:
            release_times = instance['release_times']

            f.write("RELEASE_TIME_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, s)
                for i, s in enumerate(release_times)
            ]))
            f.write("\n")

    f.write("EOF\n")
    return f.getvalue()


def get_config(*, seed, nbG, mTCV, nbC, mTTW, iTWP, tF, rP, fGNFS, mCSSD, iPLS, nbE, minP, incP, repeat=1):
    return (
        f' -maxToleratedCapacityViolation {mTCV} -nbGranular {nbG} -maxToleratedTimeWarp {mTTW}'
        f' -initialTimeWarpPenalty {iTWP} -targetFeasible {tF} -repairProbability {rP}'
        f' -nbClose {nbC} -seed {seed}'
        f' -fractionGeneratedNearest {fGNFS} -fractionGeneratedFurthest {fGNFS} -fractionGeneratedSweep {fGNFS}'
        f' -nbElite {nbE} -minimumPopulationSize {minP} -generationSize {incP}'
        f' -intensificationProbabilityLS {iPLS} -minCircleSectorSizeDegrees {mCSSD}'
        f' -doRepeatUntilTimeLimit {repeat}'
    )


def solve_static_vrptw(instance, time_limit=3600, it=20000, initial_solution=None, debug=False, config_str='', simple_init=False, use_wall_clock=True, num_keep_1=0, num_keep_2=0):
    # Prevent passing empty instances to the static solver, e.g. when
    # strategy decides to not dispatch any requests for the current epoch

    # log(f'Time limit: {time_limit}')
    if instance['coords'].shape[0] <= 1:
        return [[[], 0]]

    if instance['coords'].shape[0] <= 2:
        solution = [[1]]
        cost = validate_static_solution(instance, solution)
        return [[solution, cost]]

    str_instance = str_vrplib(instance, is_vrptw=True)

    executable = os.path.join('baselines', 'hgs_vrptw', 'genvrp')
    # On windows, we may have genvrp.exe
    if platform.system() == 'Windows' and os.path.isfile(executable + '.exe'):
        executable = executable + '.exe'
    assert os.path.isfile(executable), f"HGS executable {executable} does not exist!"
    # Call HGS solver with unlimited number of vehicles allowed and parse outputs
    # Subtract two seconds from the time limit to account for writing of the instance and delay in enforcing the time limit by HGS
    hgs_cmd = [
        executable, *
        f'-- -- -veh -1 -it {it} -nbKeepFeasibleAfterRestart {num_keep_1} -nbKeepInfeasibleAfterRestart {num_keep_2} -useWallClockTime {int(use_wall_clock)}{config_str}'.split(' ')
    ]
    t = 0
    if initial_solution is None and simple_init:
        initial_solution = [[i] for i in range(1, instance['coords'].shape[0])]
    if initial_solution is not None:
        hgs_cmd += ['-initialSolution', " ".join(map(str, to_giant_tour(initial_solution)))]

    hgs_cmd += ['-t', str(max(int(time_limit - t) - 1, 1))]
    if debug:
        log(' '.join(hgs_cmd))
    ret = []
    with subprocess.Popen(hgs_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, text=True) as p:
        p.stdin.write(str_instance)
        p.stdin.flush()
        routes = []
        for line in p.stdout:
            line = line.strip()
            if debug:
                log(line)
            # Parse only lines which contain a route
            if line.startswith('Route'):
                label, route = line.split(": ")
                route_nr = int(label.split("#")[-1])
                assert route_nr == len(routes) + 1, "Route number should be strictly increasing"
                routes.append([int(node) for node in route.split(" ")])
            elif line.startswith('Cost'):
                # End of solution
                solution = routes
                cost = int(line.split(" ")[-1].strip())
                check_cost = validate_static_solution(instance, solution)
                assert cost == check_cost, "Cost of HGS VRPTW solution could not be validated"
                ret.append([solution, cost])
                # Start next solution
                routes = []
            elif "EXCEPTION" in line:
                raise Exception("HGS failed with exception: " + line)
        assert len(routes) == 0, "HGS has terminated with incomplete solution (is the line with Cost missing?)"
    if not ret:
        log('Warning: no solution found, return simple solution')
        solution = [[i] for i in range(1, len(instance['coords']))]
        cost = validate_static_solution(instance, solution)
        return [[solution, cost]]
    return ret
