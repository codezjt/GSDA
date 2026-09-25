class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

# -gpu=2,3
# -db_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/datasets
# -baseline_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/DeepDA/save_floder/VisDA
# -save_path=/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb5295708/zjt/FixBi-main/save_path
# -padain=0