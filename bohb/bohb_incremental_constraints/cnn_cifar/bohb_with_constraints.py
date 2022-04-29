import os
import pickle
import argparse
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np

from hpbandster.optimizers.bohb import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)



parser = argparse.ArgumentParser(description='Example 5 - CNN on MNIST')
parser.add_argument('--min_budget',   type=float, help='Minimum budget of training data for training.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum budget of training data for training.',    default=10)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=32)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.', default="example")
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='./bohb_with_constraints')
parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='keras')

args=parser.parse_args()

from pytorch_worker_cnn_cifar10_incremental import PyTorchWorker as worker


# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


if args.worker:
    time.sleep(5)   # short artificial delay to make sure the nameserver is already running
    w = worker(run_id=args.run_id, host=host, timeout=120)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

time_stamp = time.time()
np.save('./temp/time_stamp', time_stamp)
constraint_results = [np.inf, 0, 0, 0, 0, 0, 0, 0]
running_results = [0, 0, 0, 0, 0, 0, 0, 0]
np.save('./temp/constraint_results', constraint_results)
np.save('./running_results', running_results)

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# Run an optimizer
bohb = BOHB(configspace=worker.get_configspace(),
            run_id=args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget,
            max_budget=args.max_budget)
res = bohb.run(n_iterations=args.n_iterations)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()