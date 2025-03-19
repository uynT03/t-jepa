import os
import torch
import numpy as np
from src.encoder import Encoder
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
import argparse
from torch.utils.data import DataLoader, TensorDataset

def load_model(args):

    args.mp_distributed = False
    args.cached_cadence=20
    args.model_ema_start = 0.996
    args.model_ema_end = 1
    args.exp_train_total_epochs = 10
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint dictionary
    checkpoint = torch.load("saved_model/jannis_model.pt", map_location=device, weights_only=False)

    context_encoder_state_dict = checkpoint["context_encoder_state_dict"]

    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)

    dataset.load()

    # Define the encoder model (ensure this matches the original architecture)
    encoder_model =  Encoder(
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

    # Load the weights into the encoder
    encoder_model.load_state_dict(context_encoder_state_dict)

    # Move to device if needed (e.g., GPU)
    encoder_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Set to eval mode (optional, for inference)
    encoder_model.eval()

    X_tensor = torch.tensor(dataset.X, dtype=torch.float32)

    # Convert dataset features (X) to a PyTorch tensor
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Process in batches and collect embeddings
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch_X = batch[0].to(device)
            batch_embeddings = encoder_model(batch_X)
            all_embeddings.append(batch_embeddings.cpu())

    # Concatenate all batch embeddings
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    embeddings_np = embeddings_tensor.numpy()

    # Save the embeddings
    np.save("datasets/jannis_embeddings.npy", embeddings_np)
    print(f"Embeddings saved with shape: {embeddings_np.shape}")

args = argparse.Namespace()
load_model(args)