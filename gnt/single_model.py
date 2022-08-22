import torch
import numpy as np
import torch.nn as nn

import os
from gnt.transformer_network import GNT
from gnt.feature_network import ResUNet

from gnt.projection import Projector
from utils import img_HWC2CHW
from gnt.render_image import render_single_image


def de_parallel(model):
    return model.module if hasattr(model, "module") else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################

class GNTWrapper(nn.Module)
   def __init__(self, net_coarse_in_feat_ch):
       super(GNTWrapper, self).__init__() 
       self.net_coarse = GNT(
            args,
            # in_feat_ch=self.args.coarse_feat_dim,
            in_feat_ch=net_coarse_in_feat_ch,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        )

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)

        # init empty module for potential future use/dev
        self.net_fine = None 

    def forward(self, rgb, camera, src_rgbs, src_cameras, depth_range):
        data["rgb"] = rgb
        data["camera"] = camera
        data["src_rgbs"] = src_rgbs
        data["src_cameras"] = src_cameras
        data["depth_range"] = depth_range

        tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
        
        ray_batch = ray_sampler.get_all()
        featmaps = model.feature_net(src_rgbs.squeeze(0).permute(0, 3, 1, 2))

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

        ''' post?
        average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

        if args.render_stride != 1:
            # gt_img = gt_img[::render_stride, ::render_stride]
            average_im = average_im[::render_stride, ::render_stride]

        # rgb_gt = img_HWC2CHW(gt_img)
        average_im = img_HWC2CHW(average_im)
        '''
        


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))
        
        # NOTE: Define networks...

        self.gntwrapper = GNTWrapper(self.args.coarse_feat_dim, self.args.fine_feat_dim, self.args.single_net, netwidth, trans_depth)

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
