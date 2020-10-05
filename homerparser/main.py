import numpy as np
import glob
import os

class HomerParser:
    def __init__(self, fpath):
        self.fpath = fpath
        self.pwm = []
        with open(fpath) as f:
            self.lines = [x.rstrip("\n") for x in f.readlines()]
        for line in self.lines:
            if line.startswith(">"):
                self.consensus, self.full_name, self.max, self.value, self._, self.meta = \
                    line.replace(">", "").split("\t")
            elif line.strip() != "":
                self.pwm.append([floor(x) for x in line.strip().split("\t")])
            if "/" in self.full_name:
                self.name = self.full_name.split("/")[0]
                if "(" in self.name:
                    self.name = self.name.split("(")[0]
            self.pwm = np.array(self.pwm)
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx < self.w:
            result = self.pwm[self.iter_idx]
            self.iter_idx += 1
            return result
        else:
            raise StopIteration
        

def parse_all_motifs(homer_dir_path):
    fpaths = glob.glob(pathname=os.path.join(homer_dir_path, 
                                                  "*.motif"), 
                            recursive=True)
    d = {x.name: x for x in [HomerParser(fp) for fp in fpaths]}
    return d

# the properties are all properties not beginning with "__"
# iterator iterates over self.values.
