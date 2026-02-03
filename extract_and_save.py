import argparse
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from utils.utils import update_args, Logger
from data.dataset import * 
from constants import MODEL_TO_NUM_LAYERS
from aggregator import get_aggregator
from extractor import Dinov3Extractor

HOME = osp.expanduser('~')

def build_dataloaders(opt, sat_mode: str = 'sat', S2D: bool = False):
    queryset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
    gallset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)
    if S2D:
        gallset = DATASET[opt.eval.dataset](mode='dro', **opt.eval)
        queryset = DATASET[opt.eval.dataset](mode=sat_mode, **opt.eval)
    query_loader = DataLoader(queryset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)
    gall_loader = DataLoader(gallset, batch_size=opt.eval.batch_size, shuffle=False, num_workers=opt.workers)
    return query_loader, gall_loader

def main():
    parser = argparse.ArgumentParser(description='DINOv3 Feature Extraction and Saving (Aligned with EM-CVGL)')
    parser.add_argument('cfg', type=str, help='yaml config path, e.g., configs/base_anyloc_D2S.yml')
    parser.add_argument('--gpu', '-g', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model_name', default=None, type=str, help='DINOv3 model name (if not provided, attempt to read from YAML model.dino_model)')
    parser.add_argument('--dinov3_local_path', default='/root/autodl-tmp/dinov3-main',type=str, help='DINOv3 local torch.hub repository path')
    parser.add_argument('--weights', type=str, help='DINOv3 weights file path (optional)')
    parser.add_argument('--agg', default='GeM', type=str, choices=['Avg', 'Max', 'GeM', 'VLAD'], help='Last layer aggregation method (if not provided, read from YAML model.aggre_type, default GeM)')
    parser.add_argument('--num_c', default=None, type=int, help='Number of cluster centers for VLAD (read from YAML model.num_c)')
    parser.add_argument('--vlad_cache_dir', type=str, help='VLAD cache directory (containing c_centers.pt, read from YAML model.vlad_cache_dir)')
    parser.add_argument('--S2D', action='store_true', help='satellite -> drone swap')
    parser.add_argument('--extend', action='store_true', help='Use extended 160k satellite gallery')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/0-pipei-dinov3/feats', help='Relative save directory name under HOME')
    parser.add_argument('--desc_layer', type=int, nargs='+', default=None, help='List of intermediate layer numbers (e.g., 22 26 30); if not provided, read from YAML or default to the last layer')
    parser.add_argument('--desc_facet', type=str, default=None, choices=['token', 'query', 'key', 'value'], help='Feature facet (if not provided, read from YAML model.desc_facet, default token)')
    parser.add_argument('--use_cls', action='store_true', help='Whether to include CLS token (only effective when facet is not token; can also be specified in YAML model.use_cls)')

    args = parser.parse_args()
    opt = update_args(args)
    gpu_str = str(opt.gpu).strip()
    run_device = 'cpu'
    if torch.cuda.is_available():
        if ',' in gpu_str or gpu_str == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            run_device = 'cuda:0'
        else:
            try:
                gpu_idx = int(gpu_str)
                run_device = f'cuda:{gpu_idx}'
            except Exception:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
                run_device = 'cuda:0'

    log_path = osp.join('outputs', opt.cfg.split('/')[-1].split('.')[0])
    os.makedirs(log_path, exist_ok=True)
    try:
        import sys, time
        sys.stdout = Logger(osp.join(log_path, f'DINOv3_Extract_{opt.eval.dataset}_{time.asctime()}.log'))
    except Exception:
        pass

    print(f"==========\nArgs:{opt}\n==========")
    def pick(val_cli, val_yaml, default=None):
        if val_cli is not None and val_cli != '':
            return val_cli
        return val_yaml if val_yaml is not None else default

    yaml_dino_model = getattr(opt.model, 'dino_model', None) if hasattr(opt, 'model') else None
    model_name = pick(args.model_name, yaml_dino_model, 'dinov3_vits16')
    if yaml_dino_model and isinstance(yaml_dino_model, str) and not str(yaml_dino_model).startswith('dinov3_'):
        if args.model_name is None:
            print(f"[Warning] YAML model.dino_model='{yaml_dino_model}' is not a DINOv3 model, switched to default/CLI: {model_name}")
    yaml_aggre_type = getattr(opt.model, 'aggre_type', None) if hasattr(opt, 'model') else None
    agg_name = pick(args.agg, yaml_aggre_type, 'GeM')

    yaml_num_c = getattr(opt.model, 'num_c', None) if hasattr(opt, 'model') else None
    yaml_vlad_cache = getattr(opt.model, 'vlad_cache_dir', None) if hasattr(opt, 'model') else None
    num_c = pick(args.num_c, yaml_num_c, 8)
    vlad_cache_dir = pick(args.vlad_cache_dir, yaml_vlad_cache, None)
    yaml_desc_layer = getattr(opt.model, 'desc_layer', None) if hasattr(opt, 'model') else None
    if args.desc_layer is not None:
        desc_layers = args.desc_layer
    elif yaml_desc_layer is not None:
        desc_layers = [yaml_desc_layer] if isinstance(yaml_desc_layer, int) else yaml_desc_layer
    else:
        desc_layers = None

    yaml_desc_facet = getattr(opt.model, 'desc_facet', None) if hasattr(opt, 'model') else None
    desc_facet = pick(args.desc_facet, yaml_desc_facet, 'token')
    yaml_use_cls = getattr(opt.model, 'use_cls', None) if hasattr(opt, 'model') else None
    use_cls = bool(args.use_cls or (yaml_use_cls is True))
    print(f"[Config] model_name={model_name}, agg={agg_name}, num_c={num_c}, vlad_cache_dir={vlad_cache_dir}, desc_layers={desc_layers}, desc_facet={desc_facet}, use_cls={use_cls}")

    agg_kwargs = {}
    if str(agg_name).lower() == 'vlad':
        agg_kwargs = dict(num_c=num_c, cache_dir=vlad_cache_dir)
    aggregator = get_aggregator(agg_name, **agg_kwargs)

    sat_mode = 'sat'
    if args.extend:
        ds_name = getattr(opt.eval, 'dataset', '') if hasattr(opt, 'eval') else ''
        if str(ds_name) == 'Feat_Single':
            sat_mode = 'sat_160k'
        else:
            print("[Info] Current evaluation dataset does not support extended 160k satellite gallery (only Feat_Single supports), reverted to regular 'sat' mode.")

    query_loader, gall_loader = build_dataloaders(opt, sat_mode=sat_mode, S2D=args.S2D)
    extractor = Dinov3Extractor(
        model_name=model_name,
        dinov3_local_path=args.dinov3_local_path,
        weights=args.weights,
        device=run_device,
        aggregator=aggregator,
        desc_layer=desc_layers, 
        desc_facet=desc_facet,
        use_cls=use_cls,
    )

    print(f'==> Extracting gallery features (Layers: {extractor.desc_layers})...')
    gall_feats_dict, gid, gall_name = extractor.extract_loader(gall_loader)
    print(f'==> Extracting query features (Layers: {extractor.desc_layers})...')
    query_feats_dict, qid, query_name = extractor.extract_loader(query_loader)
    print(HOME)
    print(args.save_dir)
    print(model_name)

    gall_view = 'sat' if not args.S2D else 'dro'
    query_view = 'dro' if not args.S2D else 'sat'

    for layer_idx in gall_feats_dict.keys():
        print(f"--- Processing Layer {layer_idx} ---")
        savedir = osp.join(HOME, args.save_dir, model_name, str(layer_idx))
        os.makedirs(savedir, exist_ok=True)
        gall_feat = gall_feats_dict[layer_idx]
        query_feat = query_feats_dict[layer_idx]
        extractor.save_view(savedir, gall_view, gall_feat, gid, gall_name)
        extractor.save_view(savedir, query_view, query_feat, qid, query_name)
    print('==> All layers extracted and saved')

if __name__ == '__main__':
    main()