from typing import List, Tuple, Optional, Union, Dict
import os
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
from constants import MODEL_TO_NUM_LAYERS

def _mkdir_if_missing(path: str):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)

class Dinov3Extractor:
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        dinov3_local_path: Optional[str] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        aggregator=None,
        desc_layer: Union[int, List[int], None] = None,
        desc_facet: str = 'token',
        use_cls: bool = False,
    ) -> None:
        self.model_name = model_name
        self.local_path = dinov3_local_path
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregator = aggregator
        self.num_layers = MODEL_TO_NUM_LAYERS.get(self.model_name, None)
        default_layers = 12
        num_layers_check = self.num_layers if self.num_layers is not None else default_layers
        if desc_layer is None:
            self.desc_layers = [num_layers_check - 1]
        elif isinstance(desc_layer, int):
            self.desc_layers = [desc_layer]
        else:
            self.desc_layers = list(desc_layer)
        self.desc_facet = str(desc_facet).lower() 
        self.use_cls = use_cls
        self.model = self._load_model()
    
        if self.num_layers is None:
             if hasattr(self.model, 'blocks'):
                 self.num_layers = len(self.model.blocks)
             else:
                 self.num_layers = 12
        self.patch_size = 16
        try:
            self.patch_size = int(getattr(self.model.patch_embed.proj, 'kernel_size')[0])
        except Exception:
            pass

        self.model = self.model.eval().to(self.device)
        self._hook_outs = {}
        self._fh_handles = []

    def _load_model(self):
        if self.local_path is not None:
            try:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True
                )
        else:
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True
                )
        return model

    @torch.inference_mode()
    def _extract_batch_feats(self, batch_x: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch_x = batch_x.to(self.device)
        target_layers = []
        for l in self.desc_layers:
            if l < 0:
                target_layers.append(self.num_layers + l)
            else:
                target_layers.append(l)
        results = {}

        if self.desc_facet == 'token':
            feats_list: List[torch.Tensor] = self.model.get_intermediate_layers(
                batch_x, n=target_layers, reshape=True, norm=True
            )
            for i, layer_idx in enumerate(self.desc_layers):
                feat_l = feats_list[i] 
                results[layer_idx] = self._aggregate_feat_batch(feat_l)
        else:
            self._hook_outs = {}
            self._fh_handles = []

            def get_hook(idx):
                def _hook(module, inputs, output):
                    self._hook_outs[idx] = output
                return _hook

            for l_idx in target_layers:
                try:
                    blk = self.model.blocks[l_idx]
                    handle = blk.attn.qkv.register_forward_hook(get_hook(l_idx))
                    self._fh_handles.append(handle)
                except Exception as e:
                    print(f"[Error] Failed to register hook for layer {l_idx}: {e}")

            _ = self.model(batch_x)

            for i, l_idx in enumerate(target_layers):
                original_layer_idx = self.desc_layers[i]
                if l_idx in self._hook_outs:
                    qkv = self._hook_outs[l_idx]
                    feat_l = self._process_qkv(qkv, batch_x)
                    results[original_layer_idx] = self._aggregate_feat_batch(feat_l)
                else:
                    raise RuntimeError(f"[Error] {l_idx} ")
            for h in self._fh_handles:
                h.remove()
            self._fh_handles = []
            self._hook_outs = {}
        return results

    def _aggregate_feat_batch(self, feat_l: torch.Tensor) -> torch.Tensor:
        out_list = []
        for i in range(feat_l.shape[0]):
            f_i = feat_l[i].detach() 
            f_i = F.normalize(f_i, p=2, dim=0)
            vec_i = self.aggregator(f_i) if self.aggregator is not None else f_i.mean(dim=(1, 2))
            vec_i = F.normalize(vec_i, dim=0) 
            out_list.append(vec_i)
        vecs = torch.stack(out_list, dim=0) 
        return vecs

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook

    def _process_qkv(self, qkv: torch.Tensor, batch_x: torch.Tensor) -> torch.Tensor:
        B, N, threeC = qkv.shape
        C = threeC // 3
        if self.desc_facet == 'query':
            res = qkv[:, :, :C]
        elif self.desc_facet == 'key':
            res = qkv[:, :, C:2 * C]
        elif self.desc_facet == 'value':
            res = qkv[:, :, 2 * C:]
        else:
            raise ValueError(f"[Error] {self.desc_facet}")

        if self.use_cls:
            tokens = res 
        else:
            tokens = res[:, 1:, :] 

        H, W = batch_x.shape[-2], batch_x.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if Hp * Wp != tokens.shape[1]:
            import math
            grid = int(math.sqrt(tokens.shape[1]))
            Hp, Wp = grid, tokens.shape[1] // grid

        feat = tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    @torch.inference_mode()
    def _extract_facet_features(self, batch_x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        old_layers = self.desc_layers
        self.desc_layers = [layer_idx]
        try:
            results = self._extract_batch_feats(batch_x)
            pass
        finally:
            self.desc_layers = old_layers
        return torch.empty(0) 

    def extract_loader(self, dataloader) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[str]]:
        feats_dict = {l: [] for l in self.desc_layers}
        ids, names = [], []
        
        for batch in tqdm(dataloader, desc='DINOv3'):
            x = batch['x'] 
            if hasattr(x, 'ndim') and x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            y = batch['y'] 
            name = batch['name'] 
            
            batch_results = self._extract_batch_feats(x)
            
            for l, v in batch_results.items():
                feats_dict[l].append(v.cpu())
                
            ids.append(y)
            names.extend(name)

        final_feats = {}
        for l, v_list in feats_dict.items():
            if len(v_list) > 0:
                final_feats[l] = torch.cat(v_list, dim=0).to(torch.float32)
            else:
                final_feats[l] = torch.empty(0)
                
        all_ids = torch.cat(ids, dim=0)
        return final_feats, all_ids, names

    @staticmethod
    def save_view(savedir: str, view: str, feats: torch.Tensor, ids: torch.Tensor, names: List[str]):
        _mkdir_if_missing(savedir)
        torch.save(feats, osp.join(savedir, f'{view}_feat'))
        torch.save(ids, osp.join(savedir, f'{view}_id'))
        torch.save(names, osp.join(savedir, f'{view}_name'))
        print(f'[success save] {view} feature save in: {savedir}')