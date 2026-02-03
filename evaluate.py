import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml
import sys
sys.path.append(os.getcwd())
from models.unsupervised_fusion import ISMS
from models.autoenc_moe import AutoEnc_MoE 

DEFAULT_ISMS_PATH = '/root/autodl-tmp/GeoAlign/outputs/base_stream2/isms_model_26_28.pth'
FEAT_ROOT_D2S = '/root/autodl-tmp/GeoAlign/feats_test/dinov3_vith16plus'
FEAT_ROOT_S2D = '/root/autodl-tmp/GeoAlign/feats_test_s2d/dinov3_vith16plus'
FUSION_LAYERS = [26, 28]
INPUT_DIM_B = 2560 
LATENT_DIM_B = 1280 
MOE_CONFIG_PATH = '/root/autodl-tmp/GeoAlign/configs/base_stream1.yml'
MOE_WEIGHT_PATH = '/root/autodl-tmp/GeoAlign/outputs/base_stream1/300_param.t'
LAYER_MAIN = 28

def load_feat(savedir, view):
    feat_path = osp.join(savedir, f'{view}_feat')
    id_path = osp.join(savedir, f'{view}_id')
    if not osp.exists(feat_path):
        raise FileNotFoundError(f"Feature not found: {feat_path}")
    feat = torch.load(feat_path, map_location='cpu').float()
    gid = torch.load(id_path, map_location='cpu')
    return feat, gid

def compute_mAP_standard(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap += precision
    ap = ap / ngood
    return ap, cmc

def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    index = np.argsort(score)[::-1]
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    ap, cmc = compute_mAP_standard(index, good_index, junk_index)
    return ap, cmc

def main():
    parser = argparse.ArgumentParser(description='A & B')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--isms_path', default=DEFAULT_ISMS_PATH, type=str, help='')
    parser.add_argument('--no_fusion', action='store_true', help='default concat Feature A  + Feature B (3840 dim) ，no fusionjust Feature A (2560 dim)')
    parser.add_argument('--mode', default='D2S', choices=['D2S', 'S2D'], help='')
    args = parser.parse_args()
    gpu_str = str(args.gpu).strip()
    device = 'cpu'
    if torch.cuda.is_available():
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            device = 'cuda:0'
        else:
            try:
                gpu_idx = int(gpu_str)
                device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                device = 'cuda:0'
    
    device = torch.device(device)
    print(f"Running on {device}")
    if args.mode == 'S2D':
        current_feat_root = FEAT_ROOT_S2D
        print(f"==> Evaluation Mode: S2D (Satellite -> Drone)")
    else:
        current_feat_root = FEAT_ROOT_D2S
        print(f"==> Evaluation Mode: D2S (Drone -> Satellite)")
    print(f"==> Feature Root: {current_feat_root}")
    use_fusion = not args.no_fusion
    if use_fusion:
        print("==> Strategy: Fusion Enabled")
    else:
        print("==> Strategy: Fusion Disabled")
    sat_B = None
    dro_B = None
    if use_fusion:
        print(f"==> Generating Feature B from {args.isms_path}...")
        if not osp.exists(args.isms_path):
            raise FileNotFoundError(f"Model file not found: {args.isms_path}. Please check the path.")
        isms = ISMS(input_dim=INPUT_DIM_B, latent_dim=LATENT_DIM_B).to(device)
        try:
            isms.load_state_dict(torch.load(args.isms_path, map_location=device))
        except RuntimeError as e:
            print(f"\n[Error] Loading State Dict failed. Detail: {e}")
            return
        isms.eval()
        sat_list, dro_list = [], []
        sat_id, dro_id = None, None
        
        print(f"  -> Loading Fusion Layers: {FUSION_LAYERS}")
        for layer in FUSION_LAYERS:
            path = osp.join(current_feat_root, str(layer))
            s_f, s_i = load_feat(path, 'sat')
            d_f, d_i = load_feat(path, 'dro')
            sat_list.append(F.normalize(s_f, dim=-1))
            dro_list.append(F.normalize(d_f, dim=-1))
            if sat_id is None: sat_id, dro_id = s_i, d_i
                
        sat_cat_b = torch.cat(sat_list, dim=-1)
        dro_cat_b = torch.cat(dro_list, dim=-1)
        
        batch_size = 256
        def get_isms_feat(data):
            outs = []
            for i in range(0, data.shape[0], batch_size):
                batch = data[i:i+batch_size].to(device)
                with torch.no_grad():
                    z, _ = isms(batch) 
                    z = F.normalize(z, dim=-1) 
                outs.append(z.cpu())
            return torch.cat(outs, dim=0)

        sat_B = get_isms_feat(sat_cat_b)
        dro_B = get_isms_feat(dro_cat_b)
        print(f"  -> Feature B (Sat/Dro) Shape: {sat_B.shape}")
    else:
        path_tmp = osp.join(current_feat_root, str(LAYER_MAIN))
        _, sat_id = load_feat(path_tmp, 'sat')
        _, dro_id = load_feat(path_tmp, 'dro')
    print("==> Generating Feature A...")
    with open(MOE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['out_dim'] = 1280
    config['model']['vec_dim'] = 2560
    moe = AutoEnc_MoE(**config['model'])
    if osp.exists(MOE_WEIGHT_PATH):
        checkpoint = torch.load(MOE_WEIGHT_PATH, map_location='cpu')
        moe.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        raise FileNotFoundError(f"Stream1 weights not found at {MOE_WEIGHT_PATH}")
    moe.to(device).eval()
    path_main = osp.join(current_feat_root, str(LAYER_MAIN))
    sat_feat_main, _ = load_feat(path_main, 'sat')
    dro_feat_main, _ = load_feat(path_main, 'dro')
    batch_size = 256
    def get_moe_feat(feat, view_mode):
        outs = []
        for i in range(0, feat.shape[0], batch_size):
            batch = feat[i:i+batch_size].to(device)
            with torch.no_grad():
                base = moe.shared_enc(batch)
                if view_mode == 'dro':
                    delta, _,_ = moe.moe_layer(base)
                    out = base + delta
                else:
                    out = base
                out = F.normalize(out, dim=-1)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)

    sat_A = get_moe_feat(sat_feat_main, 'sat')
    dro_A = get_moe_feat(dro_feat_main, 'dro')
    print(f"  -> Feature A (Sat/Dro) Shape: {sat_A.shape}")
    print(f"==> Preparing Final Features (Mode: {args.mode})...")
    if use_fusion:
        feat_sat_final = torch.cat([sat_A, sat_B], dim=-1)
        feat_sat_final = F.normalize(feat_sat_final, dim=-1)
        
        feat_dro_final = torch.cat([dro_A, dro_B], dim=-1)
        feat_dro_final = F.normalize(feat_dro_final, dim=-1)
    else:
        feat_sat_final = sat_A
        feat_dro_final = dro_A

    print(f"  -> Final Feature Dim: {feat_sat_final.shape[-1]}")
    if args.mode == 'D2S':
        gallery_feat = feat_sat_final.to(device)
        gallery_id = sat_id
        
        query_feat = feat_dro_final 
        query_id = dro_id
    else:
        print("  -> Moving Gallery (Drone features) to GPU...")
        gallery_feat = feat_dro_final.to(device)
        gallery_id = dro_id
        
        query_feat = feat_sat_final 
        query_id = sat_id
    print("==> Starting Evaluation loop...")
    gl = gallery_id.cpu().numpy()
    ql = query_id.cpu().numpy()
    CMC = torch.IntTensor(len(gl)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ql))):
        q_vec = query_feat[i].to(device) 
        ap_tmp, CMC_tmp = eval_query(q_vec, ql[i], gallery_feat, gl)
        if CMC_tmp[0] != -1:
            CMC = CMC + CMC_tmp
            ap += ap_tmp  
    AP = ap / len(ql)
    CMC = CMC.float() / len(ql)
    top1 = CMC[0]
    top5 = CMC[4] if len(CMC) > 4 else CMC[-1]
    top10 = CMC[9] if len(CMC) > 9 else CMC[-1]
    print(f'==================================================')
    print(f'Evaluation Config:')
    print(f'  Mode:   {args.mode}')
    print(f'  Fusion: {use_fusion} (Dim: {feat_sat_final.shape[-1]})')
    print(f'Results:')
    print(f'  Retrieval: top-1:  {top1:.2%}')
    print(f'             top-5:  {top5:.2%}')
    print(f'             top-10: {top10:.2%}')
    print(f'             AP:     {AP:.2%}')
    print(f'==================================================')

if __name__ == '__main__':
    main()