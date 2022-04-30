import os
import time
import atexit
from torch.utils.tensorboard import SummaryWriter


class EpochLogger:
    """
    Minimal version of spinningup's EpochLogger.
    
    References:
        https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py
        https://github.com/kakaoenterprise/JORLDY/blob/master/jorldy/manager/log_manager.py
    """
    def __init__(self, run_id=None):
        """
        Args:
            run_id (string): The name of an experiment. If ``None``, 
                , defaults to a random number.
        """
        self.run_id = run_id or str(int(time.time()))
        output_dir = os.path.join("./results", self.run_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_file = open(os.path.join(output_dir, 'progress.txt'), 'w')
        atexit.register(self.output_file.close)  # The file is automatically closed when the program exits
        
        self.writer = SummaryWriter(output_dir)
        self.first_row = True
        self.epoch_dict = dict()
        
    def log(self, **kwargs):
        """Log diagnostics"""
        for k, v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
            self.writer.add_scalar(k, v, len(self.epoch_dict[k]))
            
    def dump(self):
        """
        Write all of the diagnostics to the ouput file.
        """
        keys, vals = zip(*self.epoch_dict.items())
        self.output_file.write("\t".join(keys) + "\n")
        for row in zip(*vals):
            self.output_file.write("\t".join(map(str, row)) + "\n")
        self.output_file.flush()
