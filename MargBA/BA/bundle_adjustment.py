import os, loguru, tqdm, json

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

from MargBA.geometry import CDFLossCupy, CDFLossIndexCupy
from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.BA.transform_utils import input2cuda, random_sample_correspondence_depth

def idt_pose(device):
    idt = torch.eye(4).to(device)
    idt[2, 3] = 1.0
    return idt

def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60000000))

def initialize_pose_model(params):
    gt_pose_model = GlobalOptimizationPoseParameters(
        nfrm=params["nfrm"],
        optimizefocal=False,
        sharefocal=params['sharefocal']
    )
    # save PR Pose Model
    pr_pose_model = GlobalOptimizationPoseParameters(
        nfrm=params["nfrm"],
        optimizefocal=(not params["calibrated"]),
        sharefocal=params['sharefocal'],
        gradient_mask_exemplar=params["gradient_mask_exemplar"] if params["mapfree"] else None, # for mapfree reloc
    )

    # initialize gt_pose_model
    gt_pose_model_path = os.path.join(params["sfm_location"], "gt_pose_model.ckpt")
    if os.path.exists(gt_pose_model_path):
        gt_pose_model.load_state_dict(
            torch.load(gt_pose_model_path), strict=True
        )
    else:
        loguru.logger.warning("gt_pose_model not found, initializing with identity pose")

    # initialize pr_pose_model
    if params["stage"] == "coarse":
        pr_pose_model_path = os.path.join(
            params["sfm_location"],
            "pr_pose_model_init.ckpt"
        )
    elif params["stage"] == "fine":
        pr_pose_model_path = os.path.join(
            params["sfm_location"],
            "pr_pose_model_coarse.ckpt"
        )
    else:
        raise ValueError()

    pr_pose_model.load_state_dict(
        torch.load(
            pr_pose_model_path
        ), strict=True
    )
    return gt_pose_model, pr_pose_model

class CDFLoss(torch.nn.Module):
    def __init__(
            self,
            src_idx1,
            dst_idx2,
            w_residual,
            nfrm,
            device,
            params
    ):
        super().__init__()
        losses = dict()
        for type in params["losses"]:
            maxpx, bins, gradient_smooth = params["losses"][type]["maxpx"], params["losses"][type]["bins"], params["losses"][type]["gradient_smooth"]
            if type == "cdf_log_subgraph":
                losses[type] = CDFLossIndexCupy(
                    min=0, max=maxpx, bins=bins, src_idx1=src_idx1, dst_idx2=dst_idx2, gradient_smooth=gradient_smooth, nnodes=nfrm
                ).to(device)
            elif type == "cdf_log":
                losses[type] = CDFLossCupy(
                    min=0, max=maxpx, bins=bins, gradient_smooth=gradient_smooth
                ).to(device)
            elif type == "cdf_euclidean":
                losses[type] = CDFLossCupy(
                    min=0, max=maxpx, bins=bins, gradient_smooth=gradient_smooth
                ).to(device)
            else:
                raise ValueError("Unknown loss_type: %s" % type)
        self.w_residual = w_residual
        self.losses = losses

    def map_residual(self, residual, type):
        if type == 'cdf_log_subgraph':
            return torch.log(residual + 1)
        elif type == "cdf_log":
            return torch.log(residual + 1)
        elif type == "cdf_euclidean":
            return residual
        else:
            raise ValueError("Unknown loss_type: %s" % type)

    def forward(self, residual, weights, key):
        residual = self.map_residual(residual, key)

        if key in ["cdf_log_subgraph"]:
            loss = self.loss_wt_subgraph(residual, weights, key)
        elif key in ["cdf_log", "cdf_euclidean"]:
            loss = self.loss_wo_subgraph(residual, weights, key)
        else:
            raise ValueError("Loss type not support.")
        return loss

    def loss_wt_subgraph(self, residual, weights, key):
        cdf2D_srcidx1, cdf2D_dstidx2 = self.losses[key](residual, weights)
        cdf2D_srcidx1 = cdf2D_srcidx1 * (cdf2D_srcidx1 < 1.0).float() * self.w_residual[:, 0:1]
        cdf2D_dstidx2 = cdf2D_dstidx2 * (cdf2D_dstidx2 < 1.0).float() * self.w_residual[:, 1:2]
        cdf2D_srcidx1 = cdf2D_srcidx1[cdf2D_srcidx1 > 0]
        cdf2D_dstidx2 = cdf2D_dstidx2[cdf2D_dstidx2 > 0]
        return torch.cat([
            cdf2D_srcidx1,
            cdf2D_dstidx2
        ]).mean()

    def loss_wo_subgraph(self, residual, weights, key):
        cdf2D, _ = self.losses[key](residual, weights)
        cdf2D = cdf2D * self.w_residual * (cdf2D < 1.0).float()
        cdf2D = cdf2D[cdf2D > 0]
        return cdf2D.mean()

def sanity_check(pr_pose_model, pr_pose_model_statedict_for_sanity_check, gradient_mask):
    eps = 1e-3
    state_dict = pr_pose_model.state_dict()

    def dim_padding(x, mask):
        for i in range(x.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)
        return mask

    for key in state_dict:
        src = state_dict[key]
        dst = pr_pose_model_statedict_for_sanity_check[key]

        gradient_mask_dim_aligned = dim_padding(src, gradient_mask)

        diff = torch.abs(src - dst) * (gradient_mask_dim_aligned == 0).float()
        assert diff.max() < eps

def bundle_adjustment(
        rank,
        world_size,
        params
):
    setup(rank, world_size, params["master_port"])
    nfrm = params["nfrm"]
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    per_gpu_pair = params["per_gpu_pair"][rank]
    marker_query_map = params["marker_query_map"]
    per_gpu_weight_on_pair_dict = params["per_gpu_weight_on_pair_dict"][rank]
    gradient_mask = input2cuda(params["gradient_mask"], device=device)
    print(f"GPU {rank} process {len(per_gpu_pair)} pairs and {nfrm} nodes")

    # read data
    h5path = os.path.join(params["preprocess_location"], "{}.hdf5".format(params["hdf5name"]))
    assert os.path.exists(h5path)
    pts_src, pts_dst, depth_src, depth_dst, src_idx1, dst_idx2, visibility = random_sample_correspondence_depth(
        h5path,
        ['qry'] * nfrm if params['mapfree'] else marker_query_map,
        per_gpu_pair,
        params["sample_num"],
        min_corres_conf=params["min_corres_conf"],
        device=device
    )
    per_gpu_weight_on_pair_arr = [
        per_gpu_weight_on_pair_dict[tuple([int(src_idx1[i].item()), int(dst_idx2[i].item())])]
        for i in range(len(src_idx1))
    ]
    per_gpu_weight_on_pair_arr = input2cuda(per_gpu_weight_on_pair_arr, device=device)

    # read pose model
    gt_pose_model, pr_pose_model = initialize_pose_model(params)
    pr_pose_model.register_idx(src_idx1, dst_idx2)

    # construct loss
    cdfloss = CDFLoss(
        src_idx1,
        dst_idx2,
        per_gpu_weight_on_pair_arr,
        nfrm,
        device,
        params=params
    )

    # bundle-adjustment
    gt_pose_model = gt_pose_model.to(device)
    pr_pose_model = pr_pose_model.to(device)
    pr_pose_model_statedict_for_sanity_check = pr_pose_model.state_dict()
    pr_pose_model_ddp = DDP(pr_pose_model, device_ids=[rank])
    print(f"GPU {rank} initiate pose model...")
    optimizer = torch.optim.Adam(
        [
            {
                "params": pr_pose_model_ddp.module.get_params("extrinsic"),
                "lr": params["lr"]
            },
            {
                "params": pr_pose_model_ddp.module.get_params("intrinsic"),
                "lr": params["lr"] * params["lr_intrinsic_boost"]
            },
        ],
    )

    evaluate_function = params["evaluate_function"]

    for loss_key in params['losses']:
        for iteration in tqdm.tqdm(range(params['losses'][loss_key]['iterations']), disable=False):
            # Add barrier to ensure all processes are synchronized before evaluation
            dist.barrier()

            if np.mod(iteration, 500) == 0 and rank == 0 and evaluate_function is not None:
                result = evaluate_function(
                    pr_pose_w2c=pr_pose_model.get_pose_w2c().detach().cpu(),
                    gt_pose_w2c=gt_pose_model.get_pose_w2c().detach().cpu(),
                    marker_query_map=marker_query_map,
                    nfrm=nfrm,
                    params=params
                )
                result["loss_key"] = loss_key
                result["iteration"] = iteration
                json_string = json.dumps(result, separators=(',', ':'))
                loguru.logger.info(json_string)

                json_path = os.path.join(params["sfm_location"], "evals.json")
                if os.path.exists(json_path):
                    results = json.load(open(json_path, "r"))
                    results.append(result)
                else:
                    results = [result]

                json.dump(
                    results,
                    open(json_path, 'w'),
                    indent=4
                )

                sanity_check(pr_pose_model, pr_pose_model_statedict_for_sanity_check, gradient_mask)

            # Add another barrier to ensure all processes wait for rank 0 to finish evaluation
            dist.barrier()

            optimizer.zero_grad()
            residual, weights = pr_pose_model_ddp(
                pts_src,
                pts_dst,
                depth_src,
                depth_dst
            )
            loss = cdfloss(
                residual,
                weights,
                loss_key
            )
            loss.backward()
            optimizer.step()

    # Add barrier before saving to ensure all processes complete training
    dist.barrier()

    if rank == 0:
        torch.save(
            pr_pose_model.state_dict(),
            os.path.join(params["sfm_location"], f"pr_pose_model_{params['stage']}.ckpt")
        )

    # Final barrier to ensure model is saved before destroying process group
    dist.barrier()
    dist.destroy_process_group()
