############################
# imports
############################

import os
import socket



############################
# determine computer
############################

computers = {
    "josephus-super": "super",
    "work": "work",
    "pharma2-53": "ss",
    "work1": "backup"
}

computer = computers[socket.gethostname()]


############################
# p_chipseq
############################

bases = {"ss": "/media/daten/arnold/josephus",
         "super": "/home/josephus/Downloads",
         "work": "/media/josephus/My Book",
         "backup": "/media/josephus/mybook"}

ss = {
    "base": bases['ss'],
    "galaxy": os.path.join(bases['ss'], 'data', 'galaxy_2019_Nov'),
    "model": os.path.join(bases['ss'], 'model'),
    "genome": os.path.join(bases['ss'], 'genome'),
    "results": os.path.join(bases['ss'], 'data', 'galaxy_2019_Nov', 'results')
}

sup = {
    "base": bases['super'],
    "galaxy": os.path.join(bases['super'], 'data', 'galaxy_2019_Nov'),
    "model": os.path.join(bases['super'], 'model'),
    "genome": os.path.join(bases['super'], 'genome'),
    "results": os.path.join(bases['super'], 'data', 'galaxy_2019_Nov', 'results')
}

work = {
    "base": bases['work'],
    "archive": "/media/josephus/archive_big",
    "galaxy": os.path.join(bases['work'], 'galaxy_2019_Nov'),
    "model": os.path.join(bases['work'], 'model'),
    "genome": os.path.join(bases['work'], 'genome'),
    "results": os.path.join(bases['work'], 'galaxy_2019_Nov', 'results')
}

backup = {
    "base": bases['backup'],
    "galaxy": os.path.join(bases['backup'], 'data', 'galaxy_2019_Nov'),
    "model": os.path.join(bases['backup'], 'model'),
    "genome": os.path.join(bases['backup'], 'genome'),
    "results": os.path.join(bases['backup'], 'data', 'galaxy_2019_Nov', 'results')
}

paths = {
    "ss": ss,
    "super": sup,
    "work": work,
    "backup": backup
}

class Paths:
    def __init__(self, computer):
        self.base_dir = {'ss': '/media/daten/arnold/josephus',
                         'work': '/media/josephus/My Book',
                         'backup': '/media/josephus/mybook',
                         'super': '/home/josephus/Downloads'}[computer]
        self.gal_dir = os.path.join(self.base_dir, 'data', 'galaxy_2019_Nov')
        # in 'work' the 'data' must disappear!
        self.model_dir = os.path.join(self.base_dir, 'model')
        self.genome_dir = os.path.join(self.base_dir, 'genome')
        self.out_dir = os.path.join(self.gal_dir, 'results')
        if computer == "work":
            self.gal_dir = self.gal_dir.replace("/data/", "/")
            self.genome_dir = "/media/josephus/archive_big/genome"

p_chipseq_ = Paths(computer)
p_chipseq = paths[computer]





















