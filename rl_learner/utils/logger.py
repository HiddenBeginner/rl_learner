import atexit
import os
import time

import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Minimal version of spinningup's EpochLogger.

    References:
        https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
        https://github.com/kakaoenterprise/JORLDY/blob/master/jorldy/manager/log_manager.py
    """
    def __init__(
        self,
        output_dir=None,
        env_name='',
        agent_name='',
        run_id=None,
        verbose=True,
        use_tensorboard=True,
        use_wandb=True
    ):
        """
        Args:
            output_dir (string): The directory where results are stored
            env_name (string): The name of an environment.
            agent_name (string): The name of an agent. The name of an agent is recommended.
            run_id (string): The name of an experiment. If ``None``, defaults to a random number.
            verbose (bool): If True, print data when logging data, Default: True.
            use_tensorboard (bool): If True, use tensorboard, Default: True.
            use_wandb (bool): If True, use wandb, Default: True.
        """
        # Set the directory
        output_dir = output_dir or './results'
        self.verbose = verbose
        run_id = str(run_id) or str(int(time.time()))
        self.output_dir = os.path.join(output_dir, env_name, agent_name, run_id)

        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file = open(os.path.join(self.output_dir, 'progress.txt'), 'w')
        atexit.register(self.output_file.close)  # The file is automatically closed when the program exits

        if self.use_tensorboard:
            self.writer = SummaryWriter(self.output_dir)  # TensorBoard

        if self.use_wandb:
            wandb.init(project=env_name, group=agent_name, name=f'run_id: {run_id}')

        self.epoch_dict = dict()

    def log(self, **kwargs):
        """Log diagnostics"""
        msg = ""
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

            if self.use_tensorboard:
                self.writer.add_scalar(k, v, len(self.epoch_dict[k]))
            msg += f"{k}: {v:<6} | "

        if self.use_wandb:
            wandb.log(kwargs)

        if self.verbose:
            print(msg)

    def dump(self):
        """
        Write all of the diagnostics to the ouput file.
        """
        keys, vals = zip(*self.epoch_dict.items())
        self.output_file.write("\t".join(keys) + "\n")
        for row in zip(*vals):
            self.output_file.write("\t".join(map(str, row)) + "\n")
        self.output_file.flush()

        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            wandb.finish()
