import os, sys, torch
from torch.nn.parallel import DistributedDataParallel

class ZoeDepth:
    def __init__(self):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ZoeDepth_root = os.path.join(prj_root, 'third_party', 'ZoeDepth')
        sys.path.insert(0, ZoeDepth_root)
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        conf = get_config("zoedepth_nk", "infer")
        conf['pretrained_resource'] = "local::" + os.path.join(prj_root, 'modelzoo', 'ZoeDepth', 'ZoeD_M12_NK.pt')
        self.monodepth_estimator = build_model(conf).eval()

    def to_cuda(self, device):
        self.monodepth_estimator = self.monodepth_estimator.to(device)
    
    def inference(self, image, intr=None):
        if isinstance(self.monodepth_estimator, DistributedDataParallel):
            monodepth = self.monodepth_estimator.module.infer(image)
        else:
            monodepth = self.monodepth_estimator.infer(image)
        return monodepth, None
    
    def get_parameters(self):
        return self.monodepth_estimator.parameters()