import atexit
import os
import time

from torch.utils.tensorboard import SummaryWriter


class EpochLogger:
    """
    Minimal version of spinningup's EpochLogger.

    References:
        https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
        https://github.com/kakaoenterprise/JORLDY/blob/master/jorldy/manager/log_manager.py
    """
    def __init__(self, output_dir=None, exp_name=None, run_id=None):
        """
        Args:
            output_dir (string): A directory for saving results to. If ``None``, defaults to ``./results``
            exp_name (string): The name of a set of experiments. The name of an agent is recommended.
            run_id (string): The name of an experiment. If ``None``, defaults to a random number.
        """
        # Set the directory
        self.output_dir = output_dir or './results'  # {output_dir}
        self.output_dir = os.path.join(self.output_dir, exp_name) or self.output_dir  # {output_dir}/{exp_name}
        self.run_id = str(run_id) or str(int(time.time()))
        self.output_dir = os.path.join(self.output_dir, self.run_id)  # {output_dir}/{exp_name}/{run_id}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file = open(os.path.join(self.output_dir, 'progress.txt'), 'w')
        atexit.register(self.output_file.close)  # The file is automatically closed when the program exits

        self.writer = SummaryWriter(self.output_dir)  # TensorBoard
        self.epoch_dict = dict()

    def log(self, **kwargs):
        """Log diagnostics"""
        msg = ""
        for k, v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
            self.writer.add_scalar(k, v, len(self.epoch_dict[k]))
            msg += f"{k}: {v:<6} | "
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
