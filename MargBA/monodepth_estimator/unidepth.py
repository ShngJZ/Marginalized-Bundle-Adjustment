import os, sys, torch
import numpy as np
import PIL.Image as Image



class UniDepth:
    def __init__(self):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        UniDepth_root = os.path.join(prj_root, 'third_party', 'UniDepth')
        sys.path.insert(0, UniDepth_root)
        from unidepth.models import UniDepthV1
        self.monodepth_estimator = UniDepthV1.from_pretrained(
            "lpiccinelli/unidepth-v1-vitl14")  # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone


    def to_cuda(self, device):
        self.monodepth_estimator = self.monodepth_estimator.to(device)
        self.monodepth_estimator.eval()

    def inference(self, image, intr=None):
        predictions = self.monodepth_estimator.infer(image, intr)
        monodepth = predictions["depth"]
        if monodepth.ndim == 3:
            monodepth = monodepth.unsqueeze(1)
        return monodepth, None
