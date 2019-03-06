import torch
import torch.nn as nn

def generate(FLAGS):
    
    g_path = FLAGS.g_path

    ckpt = torch.load(g_path)

    pass
