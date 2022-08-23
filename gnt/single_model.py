import torch
import numpy as np
import torch.nn as nn

import os
from gnt.single_model_transformer_network import GNT
from gnt.feature_network import ResUNet

from gnt.single_model_render_ray import sample_along_camera_ray

from gnt.projection import Projector
from utils import img_HWC2CHW
from gnt.render_image import render_single_image

from collections import OrderedDict


def de_parallel(model):
    return model.module if hasattr(model, "module") else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################

class GNTWrapper(nn.Module):
    # self.args.coarse_feat_dim, self.args.fine_feat_dim, self.args.single_net, self.args.netwidth, self.args.trans_depth
    def __init__(self, coarse_feat_dim, fine_feat_dim, netwidth, trans_depth):
        super(GNTWrapper, self).__init__() 
         
        self.net_coarse = GNT(netwidth, trans_depth,
            # args,
            # in_feat_ch=self.args.coarse_feat_dim,
            in_feat_ch=coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            # ret_alpha=N_importance > 0, ---> FALSE
        )
        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=coarse_feat_dim, fine_out_ch=fine_feat_dim).to("cuda:0") # Single_net -> TRUE
         
        # init empty module for potential future use/dev
        self.net_fine = None 

    def forward(self, src_rgbs, ray_sampler_H, ray_sampler_W, ray_o, ray_d, camera, depth_range, src_cameras, chunk_size, N_samples, N_importance, render_stride):
        featmaps = self.feature_net(src_rgbs)
        '''
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            ret_alpha=ret_alpha,
            single_net=single_net,
        )
        '''
        ### original inputs ray_sampler=ray_sampler, ray_batch=ray_batch, model=model, projector=projector, chunk_size=args.chunk_size, N_samples=args.N_samples, inv_uniform=args.inv_uniform, det=True, N_importance=args.N_importance, white_bkgd=args.white_bkgd, render_stride=render_stride, featmaps=featmaps, ret_alpha=ret_alpha, single_net=single_net

        ### New Inputs     : ray_sampler.H, ray_sampler.W, ray_batch["ray_o"], camera, depth_range, src_rgbs, src_cameras, NO_MODEL, NO_PROJECTOR(?), chunk_size, N_samples, N_importance, render_stride,  

        all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])

        # N_rays = ray_batch["ray_o"].shape[0]
        N_rays = ray_o.shape[0]

        # TODO after porting render_rays in to this forward function rewrite this for loop
        for i in range(0, N_rays, chunk_size):
            chunk = OrderedDict()
            '''
            for k in ray_batch:
                if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                    chunk[k] = ray_batch[k]
                elif ray_batch[k] is not None:
                    chunk[k] = ray_batch[k][i : i + chunk_size]
                else:
                    chunk[k] = None
            '''

            chunk["camera"] = camera
            chunk["depth_range"] = depth_range
            chunk["src_rgbs"] = src_rgbs
            chunk["src_cameras"] = src_cameras

            '''
            ret = render_rays(
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            N_importance=N_importance,
            det=det,
            white_bkgd=white_bkgd,
            ret_alpha=ret_alpha,
            single_net=single_net,
            )
            '''

            ret = {"outputs_coarse": None, "outputs_fine": None}
            # ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"]

            # pts: [N_rays, N_samples, 3]
            # z_vals: [N_rays, N_samples]
            pts, z_vals = sample_along_camera_ray(
                ray_o=ray_o,
                ray_d=ray_d,
                depth_range=depth_range,
                N_samples=N_samples,
            )

            N_rays, N_samples = pts.shape[:2]

            # TODO: Inference gets to here before erroring on projector.compute being undefined
            # it was originally passed in as a function argument.

            rgb_feat, ray_diff, mask = projector.compute(
                pts,
                ray_batch["camera"],
                ray_batch["src_rgbs"],
                ray_batch["src_cameras"],
                featmaps=featmaps[0],
            )

            rgb = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)
            if ret_alpha:
                rgb, weights = rgb[:, 0:3], rgb[:, 3:]
                depth_map = torch.sum(weights * z_vals, dim=-1)
            else:
                weights = None
                depth_map = None
            ret["outputs_coarse"] = {"rgb": rgb, "weights": weights, "depth": depth_map}

            if N_importance > 0:
                # detach since we would like to decouple the coarse and fine networks
                weights = ret["outputs_coarse"]["weights"].clone().detach()  # [N_rays, N_samples]
                pts, z_vals = sample_fine_pts(
                    inv_uniform, N_importance, det, N_samples, ray_batch, weights, z_vals
                )

                rgb_feat_sampled, ray_diff, mask = projector.compute(
                    pts,
                    ray_batch["camera"],
                    ray_batch["src_rgbs"],
                    ray_batch["src_cameras"],
                    featmaps=featmaps[1],
                )

                # TODO: Include pixel mask in ray transformer
                # pixel_mask = (
                #     mask[..., 0].sum(dim=2) > 1
                # )  # [N_rays, N_samples]. should at least have 2 observations

                if single_net:
                    rgb = model.net_coarse(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
                else:
                    rgb = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)
                rgb, weights = rgb[:, 0:3], rgb[:, 3:]
                depth_map = torch.sum(weights * z_vals, dim=-1)
                ret["outputs_fine"] = {"rgb": rgb, "weights": weights, "depth": depth_map}


            # handle both coarse and fine outputs
            # cache chunk results on cpu
            if i == 0:
                for k in ret["outputs_coarse"]:
                    if ret["outputs_coarse"][k] is not None:
                        all_ret["outputs_coarse"][k] = []

                if ret["outputs_fine"] is None:
                    all_ret["outputs_fine"] = None
                else:
                    for k in ret["outputs_fine"]:
                        if ret["outputs_fine"][k] is not None:
                            all_ret["outputs_fine"][k] = []

            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

            if ret["outputs_fine"] is not None:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k].append(ret["outputs_fine"][k].cpu())

        rgb_strided = torch.ones(ray_sampler_H, ray_sampler_W, 3)[::render_stride, ::render_stride, :]
        # merge chunk results and reshape
        for k in all_ret["outputs_coarse"]:
            if k == "random_sigma":
                continue
            tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape(
                (rgb_strided.shape[0], rgb_strided.shape[1], -1))
        
            all_ret["outputs_coarse"][k] = tmp.squeeze()

        # TODO: if invalid: replace with white
        # all_ret["outputs_coarse"]["rgb"][all_ret["outputs_coarse"]["mask"] == 0] = 1.0
        if all_ret["outputs_fine"] is not None:
            for k in all_ret["outputs_fine"]:
                if k == "random_sigma":
                    continue
                tmp = torch.cat(all_ret["outputs_fine"][k], dim=0).reshape(
                    (rgb_strided.shape[0], rgb_strided.shape[1], -1))

                all_ret["outputs_fine"][k] = tmp.squeeze()

        return all_ret
        


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        
        # NOTE: Define networks...

        self.gntwrapper = GNTWrapper(self.args.coarse_feat_dim, self.args.fine_feat_dim, self.args.netwidth, self.args.trans_depth)

        '''
        # create coarse GNT
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        ).to(device)
        
        # single_net - trains single network which can be used for both coarse and fine sampling
        if args.single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)
        '''

        # optimizer and learning rate scheduler
        learnable_params = list(self.gntwrapper.net_coarse.parameters())
        learnable_params += list(self.gntwrapper.feature_net.parameters())
        if self.gntwrapper.net_fine is not None:
            learnable_params += list(self.gntwrapper.net_fine.parameters())

        if self.gntwrapper.net_fine is not None:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.gntwrapper.net_coarse.parameters()},
                    {"params": self.gntwrapper.net_fine.parameters()},
                    {"params": self.gntwrapper.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.gntwrapper.net_coarse.parameters()},
                    {"params": self.gntwrapper.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.gntwrapper.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.gntwrapper.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.gntwrapper.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.gntwrapper.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            if self.gntwrapper.net_fine is not None:
                self.gntwrapper.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.gntwrapper.net_fine, device_ids=[args.local_rank], output_device=args.local_rank
                )

        if args.onnx and args.coreml:
            self.gntwrapper.onnx_export()
            self.gntwrapper.coreml_export()
            exit()

        elif args.onnx:
            self.gntwrapper.onnx_export()
            exit()

        elif args.coreml:
            self.gntwrapper.coreml_export()
            exit()

    def switch_to_eval(self):
        self.gntwrapper.net_coarse.eval()
        self.gntwrapper.feature_net.eval()
        if self.gntwrapper.net_fine is not None:
            self.gntwrapper.net_fine.eval()

    def switch_to_train(self):
        self.gntwrapper.net_coarse.train()
        self.gntwrapper.feature_net.train()
        if self.gntwrapper.net_fine is not None:
            self.gntwrapper.net_fine.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            # "net_coarse": de_parallel(self.net_coarse).state_dict(),
            # "feature_net": de_parallel(self.feature_net).state_dict(),
            "gntwrapper": de_parallel(self.gntwrapper).state_dict()
        }

        ''' not needed if self.net_fine is in gntwrapper
        if self.gntwrapper.net_fine is not None:
            to_save["net_fine"] = de_parallel(self.gntwrapper.net_fine).state_dict()
        '''
        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        # print(to_load["net_coarse"].keys())
        # exit()
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])
        
        try:
            self.gntwrapper.load_state_dict(to_load["gntwrapper"])
        except:
            self.gntwrapper.feature_net.load_state_dict(to_load["feature_net"])
        
            intermediary_sd = self.gntwrapper.net_coarse.state_dict()

            '''
            for item in intermediary_sd.keys():
                print(item, intermediary_sd[item].shape)
            print()
            for item in to_load["net_coarse"].keys():
                print(item, to_load["net_coarse"][item].shape)
            '''

            # i = 0
            for item in intermediary_sd.keys():
                # NOTE: copy weights into model with programtic name edits...            
                if "view_trans" in item:
                    new_name = item.replace("view_trans", "view_crosstrans")
                    intermediary_sd[item] = to_load["net_coarse"][new_name]
                elif "ray_trans" in item:
                    new_name = item.replace("ray_trans", "view_selftrans")
                    intermediary_sd[item] = to_load["net_coarse"][new_name]
                else:
                    intermediary_sd[item] = to_load["net_coarse"][item]
            

            self.gntwrapper.net_coarse.load_state_dict(intermediary_sd)

            torch.save(self.gntwrapper.net_coarse.state_dict(), "./converted_model.pth")
            # self.net_coarse.load_state_dict(to_load["net_coarse"])
            # self.feature_net.load_state_dict(to_load["feature_net"])
            '''
            if self.gntwrapper.net_fine is not None and "net_fine" in to_load.keys():
                self.gntwrapper.net_fine.load_state_dict(to_load["net_fine"])
            '''

    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step
