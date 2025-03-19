import os, copy
import random
import argparse
import numpy as np
from tabulate import tabulate
import wandb
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.encode_utils import encode_data
import src.utils.idr_torch as idr_torch  # JEAN-ZAY

from src.encoder import Encoder
from src.predictors import Predictors
from src.torch_dataset import TorchDataset
from src.train import Trainer
from src.mask import MaskCollator
from src.configs import build_parser
from src.utils.log_utils import make_job_name
from src.utils.log_utils import print_args
from src.utils.checkpointer import EarlyStopCounter
from src.utils.train_utils import init_weights
from src.utils.optim_utils import init_optim

from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP


# def main(args):
#
#     if args.mp_distributed:
#         dist.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             world_size=idr_torch.size,
#             rank=idr_torch.rank,
#         )
#
#     ema_start = args.model_ema_start
#
#     ema_end = args.model_ema_end
#     num_epochs = args.exp_train_total_epochs
#     ipe_scale = args.exp_ipe_scale
#
#     dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
#     args.is_batchlearning = args.batch_size != -1
#     args.iteration = 0
#     start_epoch = 0
#     if args.test:
#         args.mock = True
#
#     if (not args.mp_distributed) or (args.mp_distributed and idr_torch.local_rank == 0):
#         if args.verbose:
#             print_args(args)
#
#     if args.random:
#         args.torch_seed = np.random.randint(0, 100000)
#         args.np_seed = np.random.randint(0, 100000)
#
#     torch.manual_seed(args.torch_seed)
#     np.random.seed(args.np_seed)
#     random.seed(args.np_seed)
#
#     jobname = make_job_name(args)
#
#     print(tabulate(vars(args).items(), tablefmt="fancy_grid"))
#
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device)
#     dataset.load()
#     args.test_size = 0
#     train_torchdataset = TorchDataset(
#         dataset=dataset,
#         mode="train",
#         kwargs=args,
#         device=device,
#         preprocessing=encode_data,
#     )
#
#     context_encoder = Encoder(
#         idx_num_features=dataset.num_features,
#         cardinalities=dataset.cardinalities,
#         hidden_dim=args.model_dim_hidden,
#         num_layers=args.model_num_layers,
#         num_heads=args.model_num_heads,
#         p_dropout=args.model_dropout_prob,
#         layer_norm_eps=args.model_layer_norm_eps,
#         gradient_clipping=args.exp_gradient_clipping,
#         feature_type_embedding=args.model_feature_type_embedding,
#         feature_index_embedding=args.model_feature_index_embedding,
#         dim_feedforward=args.model_dim_feedforward,
#         device=device,
#         args=args,
#     )
#
#     predictors = Predictors(
#         pred_type=args.pred_type,
#         hidden_dim=args.model_dim_hidden,
#         pred_embed_dim=args.pred_embed_dim,
#         num_features=dataset.D,
#         num_layers=args.pred_num_layers,
#         num_heads=args.pred_num_heads,
#         p_dropout=args.pred_p_dropout,
#         layer_norm_eps=args.pred_layer_norm_eps,
#         activation=args.pred_activation,
#         device=device,
#         cardinalities=dataset.cardinalities,
#         pred_dim_feedforward=args.pred_dim_feedforward,
#     )
#
#     for m in context_encoder.modules():
#         init_weights(m, init_type=args.init_type)
#
#     if args.pred_type == "mlp":
#         for pred in predictors.predictors:
#             for m in pred.modules():
#                 init_weights(m, init_type=args.init_type)
#
#     target_encoder = copy.deepcopy(context_encoder)
#
#     context_encoder.to(device)
#     target_encoder.to(device)
#     predictors.to(device)
#
#     scaler = GradScaler(enabled=args.model_amp)
#     if args.model_amp:
#         print(f"Initialized gradient scaler for Automatic Mixed Precision.")
#
#     early_stop_counter = EarlyStopCounter(
#         args, jobname, args.data_set, device=device, is_distributed=False
#     )
#
#     mask_collator = MaskCollator(
#         args.mask_allow_overlap,
#         args.mask_min_ctx_share,
#         args.mask_max_ctx_share,
#         args.mask_min_trgt_share,
#         args.mask_max_trgt_share,
#         args.mask_num_preds,
#         args.mask_num_encs,
#         dataset.D,
#         dataset.cardinalities,
#     )
#
#     dataloader = DataLoader(
#         dataset=train_torchdataset,
#         batch_size=args.batch_size,
#         num_workers=args.data_loader_nprocs,
#         collate_fn=mask_collator,
#         pin_memory=args.pin_memory,
#         drop_last=False,
#     )
#
#     ipe = len(dataloader)
#
#     (optimizer, scheduler, weightdecay_scheduler) = init_optim(
#         context_encoder,
#         predictors,
#         ipe,
#         args.exp_start_lr,
#         args.exp_lr,
#         args.exp_warmup,
#         args.exp_train_total_epochs,
#         args.exp_weight_decay,
#         args.exp_final_weight_decay,
#         args.exp_final_lr,
#         args.exp_ipe_scale,
#         args.exp_scheduler,
#         args.exp_weight_decay_scheduler,
#     )
#
#     momentum_scheduler = (
#         ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
#         for i in range(int(ipe * num_epochs * ipe_scale) + 1)
#     )
#
#     if args.load_from_checkpoint:
#         if os.path.isfile(args.load_path):
#             (
#                 context_encoder,
#                 predictors,
#                 target_encoder,
#                 optimizer,
#                 scaler,
#                 scheduler,
#                 weightdecay_scheduler,
#                 start_epoch,
#             ) = early_stop_counter.load_model(
#                 load_pth=args.load_path,
#                 context_encoder=context_encoder,
#                 predictor=predictors,
#                 target_encoder=target_encoder,
#                 optimizer=optimizer,
#                 scaler=scaler,
#                 scheduler=scheduler,
#                 weightdecay_scheduler=weightdecay_scheduler,
#             )
#             for _ in range(start_epoch * ipe):
#                 next(momentum_scheduler)
#                 mask_collator.step()
#         else:
#             print(
#                 "Tried loading from checkpoint,"
#                 " but provided path does not exist."
#                 " Starting training from scratch."
#             )
#
#         for p in target_encoder.parameters():
#             p.requires_grad = False
#
#     trainer = Trainer(
#         args=args,
#         start_epoch=start_epoch,
#         context_encoder=context_encoder,
#         target_encoder=target_encoder,
#         predictors=predictors,
#         scheduler=scheduler,
#         weightdecay_scheduler=weightdecay_scheduler,
#         early_stop_counter=early_stop_counter,
#         momentum_scheduler=momentum_scheduler,
#         optimizer=optimizer,
#         scaler=scaler,
#         torch_dataset=train_torchdataset,
#         dataloader=dataloader,
#         distributed_args=None,
#         device=device,
#         probe_cadence=args.probe_cadence,
#         probe_model=args.probe_model,
#     )
#
#     print("Starting training...")
#     trainer.train()

def main(args):
    args.mp_distributed = False
    args.cached_cadence=2
    args.model_ema_start = 0.996
    args.model_ema_end = 1
    args.exp_train_total_epochs = 2
    args.exp_ipe_scale = 1.0
    args.data_set = "jannis"
    args.batch_size = 512
    args.test = False
    args.mock = False
    args.verbose = False
    args.random = False
    args.model_dim_hidden = 64
    args.model_num_layers = 16
    args.model_num_heads = 2
    args.model_dropout_prob = 0.0018542257314766695
    args.model_layer_norm_eps = 1.0e-5
    args.exp_gradient_clipping = 1.0
    args.model_feature_type_embedding = False
    args.model_feature_index_embedding = False
    args.model_dim_feedforward = 64
    args.pred_type = "transformer"
    args.pred_embed_dim = 16
    args.pred_num_layers = 16
    args.pred_num_heads = 4
    args.pred_p_dropout = 0.0019307456645321797
    args.pred_layer_norm_eps = 1e-5
    args.pred_activation = "relu"
    args.pred_dim_feedforward = 256
    args.init_type= "kaiming"
    args.model_amp = False
    args.mask_allow_overlap = False
    args.mask_min_ctx_share = 0.13628349247854785
    args.mask_max_ctx_share = 0.36819012681604135
    args.mask_min_trgt_share = 0.1556321308543076
    args.mask_max_trgt_share = 0.6222278244105446
    args.mask_num_preds = 4
    args.mask_num_encs = 1
    args.data_loader_nprocs = 0
    args.pin_memory = False
    args.model_act_func = "relu"
    args.exp_start_lr = 0
    args.exp_lr = 1e-4
    args.exp_warmup = 10
    args.exp_weight_decay = 0
    args.exp_final_weight_decay = 0
    args.exp_final_lr = 0
    args.exp_ipe_scale = 1.0
    args.exp_scheduler = True
    args.exp_weight_decay_scheduler = True
    args.load_from_checkpoint = False
    args.load_path = None
    args.exp_cache_cadence = 1
    args.exp_val_cache_cadence = 1
    args.data_path = "./datasets"
    args.probe_model = "linear_probe"
    args.probe_cadence = 20
    args.patience = 100
    args.init_type = "trunc_normal"
    args.log_tensorboard = False
    args.tjepa_random = False
    args.torch_seed = 42
    args.n_cls_tokens = 1
    args.mock = False
    args.res_dir = "./"
    args.test_size_ratio = 0.0
    args.val_size_ratio = 0.0
    args.random_state = 42
    args.full_dataset_cuda = False
    args.verbose = False
    args.test = False
    args.exo_n_runs = 1
    args.exp_device = None
    args.np_seed = 42
    args.random = False
    args.val_batch_size = -1
    args.exp_patience = 10
    args.exp_cadence_type = "improvement"
    args.exp_n_runs = 1
    args.mp_nodes = 1
    args.mp_gpus = 1
    args.mp_nr = 0
    args.using_embedding = False
    args.model_dtype = "float32"

    if args.mp_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.size,
            rank=idr_torch.rank,
        )

    ema_start = args.model_ema_start
    ema_end = args.model_ema_end
    num_epochs = args.exp_train_total_epochs
    ipe_scale = args.exp_ipe_scale

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
    args.is_batchlearning = args.batch_size != -1
    args.iteration = 0
    start_epoch = 0
    if args.test:
        args.mock = True

    if (not args.mp_distributed) or (args.mp_distributed and idr_torch.local_rank == 0):
        if args.verbose:
            print_args(args)

    if args.random:
        args.torch_seed = np.random.randint(0, 100000)
        args.np_seed = np.random.randint(0, 100000)

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    random.seed(args.np_seed)

    jobname = make_job_name(args)

    print(tabulate(vars(args).items(), tablefmt="fancy_grid"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataset.load()
    args.test_size = 0
    train_torchdataset = TorchDataset(
        dataset=dataset,
        mode="train",
        kwargs=args,
        device=device,
        preprocessing=encode_data,
    )

    context_encoder = Encoder(
        idx_num_features=dataset.num_features,
        cardinalities=dataset.cardinalities,
        hidden_dim=args.model_dim_hidden,
        num_layers=args.model_num_layers,
        num_heads=args.model_num_heads,
        p_dropout=args.model_dropout_prob,
        layer_norm_eps=args.model_layer_norm_eps,
        gradient_clipping=args.exp_gradient_clipping,
        feature_type_embedding=args.model_feature_type_embedding,
        feature_index_embedding=args.model_feature_index_embedding,
        dim_feedforward=args.model_dim_feedforward,
        device=device,
        args=args,
    )

    predictors = Predictors(
        pred_type=args.pred_type,
        hidden_dim=args.model_dim_hidden,
        pred_embed_dim=args.pred_embed_dim,
        num_features=dataset.D,
        num_layers=args.pred_num_layers,
        num_heads=args.pred_num_heads,
        p_dropout=args.pred_p_dropout,
        layer_norm_eps=args.pred_layer_norm_eps,
        activation=args.pred_activation,
        device=device,
        cardinalities=dataset.cardinalities,
        pred_dim_feedforward=args.pred_dim_feedforward,
    )

    for m in context_encoder.modules():
        init_weights(m, init_type=args.init_type)

    if args.pred_type == "mlp":
        for pred in predictors.predictors:
            for m in pred.modules():
                init_weights(m, init_type=args.init_type)

    target_encoder = copy.deepcopy(context_encoder)

    context_encoder.to(device)
    target_encoder.to(device)
    predictors.to(device)

    scaler = GradScaler(enabled=args.model_amp)
    if args.model_amp:
        print(f"Initialized gradient scaler for Automatic Mixed Precision.")

    early_stop_counter = EarlyStopCounter(
        args, jobname, args.data_set, device=device, is_distributed=False
    )

    mask_collator = MaskCollator(
        args.mask_allow_overlap,
        args.mask_min_ctx_share,
        args.mask_max_ctx_share,
        args.mask_min_trgt_share,
        args.mask_max_trgt_share,
        args.mask_num_preds,
        args.mask_num_encs,
        dataset.D,
        dataset.cardinalities,
    )

    dataloader = DataLoader(
        dataset=train_torchdataset,
        batch_size=args.batch_size,
        num_workers=args.data_loader_nprocs,
        collate_fn=mask_collator,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    ipe = len(dataloader)

    (optimizer, scheduler, weightdecay_scheduler) = init_optim(
        context_encoder,
        predictors,
        ipe,
        args.exp_start_lr,
        args.exp_lr,
        args.exp_warmup,
        args.exp_train_total_epochs,
        args.exp_weight_decay,
        args.exp_final_weight_decay,
        args.exp_final_lr,
        args.exp_ipe_scale,
        args.exp_scheduler,
        args.exp_weight_decay_scheduler,
    )

    momentum_scheduler = (
        ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    if args.load_from_checkpoint:
        if os.path.isfile(args.load_path):
            (
                context_encoder,
                predictors,
                target_encoder,
                optimizer,
                scaler,
                scheduler,
                weightdecay_scheduler,
                start_epoch,
            ) = early_stop_counter.load_model(
                load_pth=args.load_path,
                context_encoder=context_encoder,
                predictor=predictors,
                target_encoder=target_encoder,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                weightdecay_scheduler=weightdecay_scheduler,
            )
            for _ in range(start_epoch * ipe):
                next(momentum_scheduler)
                mask_collator.step()
        else:
            print(
                "Tried loading from checkpoint,"
                " but provided path does not exist."
                " Starting training from scratch."
            )

        for p in target_encoder.parameters():
            p.requires_grad = False

    trainer = Trainer(
        args=args,
        start_epoch=start_epoch,
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        scheduler=scheduler,
        weightdecay_scheduler=weightdecay_scheduler,
        early_stop_counter=early_stop_counter,
        momentum_scheduler=momentum_scheduler,
        optimizer=optimizer,
        scaler=scaler,
        torch_dataset=train_torchdataset,
        dataloader=dataloader,
        distributed_args=None,
        device=device,
        probe_cadence=args.probe_cadence,
        probe_model=args.probe_model,
    )

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    wandb.init(mode="offline")
    # parser = build_parser()
    # args = parser.parse_args()
    args = argparse.Namespace()
    main(args)
