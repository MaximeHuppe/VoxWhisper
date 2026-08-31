# src/models/vox_whisper.py
import torch
import torch.nn as nn
from .encoder import Encoder, ResidualBlock
from .attention import CrossVolumeAttention, PromptDecoder
from .decoder import Decoder


_PIPELINE_STEPS = {
    "t1_encoder": "Step 1  T1 visual encoder",
    "t2_encoder": "Step 1  T2 visual encoder",
    "cross_volume_attention": "Step 2  Spatial alignment (cross-volume MHA)",
    "prompt_decoder": "Step 3  Semantic alignment (prompt decoder)",
    "decoder": "Step 4  Hierarchical decoder + deep supervision",
}


def _numel_stats(module):
    """Return (n_params, n_trainable, n_buffers, nbytes) for a module tree."""
    n_params = 0
    n_trainable = 0
    nbytes = 0
    for p in module.parameters():
        n_params += p.numel()
        nbytes += p.numel() * p.element_size()
        if p.requires_grad:
            n_trainable += p.numel()
    n_buffers = 0
    for b in module.buffers():
        n_buffers += b.numel()
        nbytes += b.numel() * b.element_size()
    return n_params, n_trainable, n_buffers, nbytes


def _fmt_count(n):
    return f"{n:,}"


def _fmt_bytes(n):
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def _fmt_shape(shape):
    return "[" + ", ".join(str(int(d)) for d in shape) + "]"


def _capture(obj):
    """Snapshot nested tensor shapes/nbytes without keeping the tensors alive."""
    if torch.is_tensor(obj):
        return {
            "shape": tuple(int(s) for s in obj.shape),
            "nbytes": obj.numel() * obj.element_size(),
        }
    if isinstance(obj, (list, tuple)):
        return [_capture(x) for x in obj]
    return None


def _iter_captured(captured, prefix="out"):
    if captured is None:
        return
    if isinstance(captured, dict) and "shape" in captured:
        yield prefix, captured["shape"], captured["nbytes"]
        return
    if isinstance(captured, list):
        if len(captured) == 1:
            yield from _iter_captured(captured[0], prefix)
            return
        for i, item in enumerate(captured):
            yield from _iter_captured(item, f"{prefix}[{i}]")


def _output_labels(module, n_tensors):
    name = type(module).__name__
    if name == "Encoder" and n_tensors == 4:
        return ["bottleneck", "skip3", "skip2", "skip1"]
    if name in {"Decoder", "VoxWhisper"} and n_tensors == 3:
        return ["pred ×1/4", "pred ×1/2", "pred ×1"]
    if name == "MultiheadAttention" and n_tensors == 2:
        return ["attn_out", "attn_weights"]
    if name == "StageVLFusionBlock" and n_tensors == 2:
        return ["modulated", "mask_logits"]
    return None


def _print_table(headers, rows, aligns=None):
    """Print a left/right-aligned ASCII table. ``rows`` are sequences of strings."""
    if not headers:
        return
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    aligns = aligns or ["left"] * len(headers)

    def fmt_row(vals):
        parts = []
        for i, v in enumerate(vals):
            parts.append(v.rjust(widths[i]) if aligns[i] == "right" else v.ljust(widths[i]))
        return "  " + "  ".join(parts)

    print(fmt_row(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in str_rows:
        print(fmt_row(row))


def _count_types(module):
    return {
        "Conv3d": sum(1 for m in module.modules() if isinstance(m, nn.Conv3d)),
        "Linear": sum(1 for m in module.modules() if isinstance(m, nn.Linear)),
        "MultiheadAttention": sum(
            1 for m in module.modules() if isinstance(m, nn.MultiheadAttention)
        ),
        "ResidualBlock": sum(1 for m in module.modules() if isinstance(m, ResidualBlock)),
        "InstanceNorm3d": sum(1 for m in module.modules() if isinstance(m, nn.InstanceNorm3d)),
        "LayerNorm": sum(1 for m in module.modules() if isinstance(m, nn.LayerNorm)),
    }


class VoxWhisper(nn.Module):
    """
    VoxWhisper: A 3D Multi-Modal, Language-Grounded Volumetric Segmentation Model.
    Fuses unregistered T1 and T2 structural MRIs, aligns them with clinical prompts,
    and reconstructs prompt-conditioned segmentation masks in the T1 coordinate space.
    """

    def __init__(
        self,
        input_channels=1,
        text_dim=768,
        embed_dim=128,
        channels=None,
        strides=None,
        kernel_sizes=None,
        paddings=None,
        num_resblocks=None,
        num_heads=4,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128]
        if strides is None:
            strides = [2, 2, 2]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if paddings is None:
            paddings = [1, 1, 1]
        if num_resblocks is None:
            num_resblocks = [1, 1, 1]

        # 1. Visual Encoders (T1 Branch & T2 Branch)
        encoder_kwargs = dict(
            input_channels=input_channels,
            channels=channels,
            strides=strides,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            num_resblocks=num_resblocks,
        )
        self.t1_encoder = Encoder(**encoder_kwargs)
        self.t2_encoder = Encoder(**encoder_kwargs)

        # 2. Visual-to-Visual Alignment (Fuses T2 into T1 coordinate space)
        self.cross_volume_attention = CrossVolumeAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # 3. Language-to-Visual Alignment
        self.prompt_decoder = PromptDecoder(
            text_dim=text_dim, embed_dim=embed_dim, num_heads=num_heads
        )

        # 4. Hierarchical Decoder with Channel Modulation & Deep Supervision
        self.decoder = Decoder(channels=channels, query_dim=embed_dim)

    @classmethod
    def from_config(cls, config):
        """Construct the model from the ``model`` section of a YAML config."""
        model_cfg = config["model"]
        enc = model_cfg["encoder"]
        return cls(
            input_channels=model_cfg["input_channels"],
            text_dim=model_cfg["text_dim"],
            embed_dim=model_cfg["embed_dim"],
            channels=enc["channels"],
            strides=enc["strides"],
            kernel_sizes=enc["kernel_sizes"],
            paddings=enc["paddings"],
            num_resblocks=enc["num_resblocks"],
            num_heads=model_cfg["num_heads"],
        )

    def print_summary(self, config=None, *, max_depth=4):
        """Print parameter counts, memory size, and a per-module breakdown."""
        total, trainable, n_buffers, nbytes = _numel_stats(self)
        frozen = total - trainable
        p0 = next(self.parameters(), None)
        device = str(p0.device) if p0 is not None else "n/a"
        dtype = str(p0.dtype) if p0 is not None else "n/a"

        enc = self.t1_encoder
        channels = [enc.stem[0].out_channels] + [
            stage.transition[0].out_channels for stage in enc.stages
        ]
        strides = [stage.transition[0].stride[0] for stage in enc.stages]
        text_dim = self.prompt_decoder.text_projection.in_features
        embed_dim = self.prompt_decoder.embed_dim
        num_heads = self.cross_volume_attention.mha.num_heads
        types = _count_types(self)

        io_map = None
        dummy = None
        if config is not None:
            io_map, dummy = self._probe_forward(config)

        width = 100
        bar = "=" * width
        thin = "-" * width
        print(bar)
        print("VoxWhisper")
        print(bar)
        print(f"  {'Device':<24} {device}")
        print(f"  {'Parameter dtype':<24} {dtype}")
        print(f"  {'Encoder channels':<24} {channels}")
        print(
            f"  {'Encoder strides':<24} {strides}  "
            f"(spatial downsample ×{2 ** len(strides)})"
        )
        print(f"  {'Text dim → embed dim':<24} {text_dim} → {embed_dim}")
        print(f"  {'Attention heads':<24} {num_heads}")
        print(
            f"  {'Layer counts':<24} "
            f"{types['Conv3d']} Conv3d  |  {types['Linear']} Linear  |  "
            f"{types['MultiheadAttention']} MHA  |  {types['ResidualBlock']} ResidualBlock"
        )
        print(
            f"  {'':<24} "
            f"{types['InstanceNorm3d']} InstanceNorm3d  |  {types['LayerNorm']} LayerNorm"
        )
        print()
        print(f"  {'Total parameters':<24} {_fmt_count(total)}")
        print(f"  {'  trainable':<24} {_fmt_count(trainable)}")
        print(f"  {'  frozen':<24} {_fmt_count(frozen)}")
        print(f"  {'Buffers':<24} {_fmt_count(n_buffers)}")
        print(f"  {'Parameter memory':<24} {_fmt_bytes(nbytes)}  (weights + buffers)")
        print(thin)
        print("  Pipeline")
        print(thin)
        pipe_rows = []
        for name, child in self.named_children():
            n, _, _, child_bytes = _numel_stats(child)
            share = (100.0 * n / total) if total else 0.0
            out_str = ""
            if io_map is not None and id(child) in io_map:
                _, out_c = io_map[id(child)]
                out_str = " | ".join(
                    _fmt_shape(shape) for _, shape, _ in _iter_captured(out_c)
                )
            pipe_rows.append(
                (
                    _PIPELINE_STEPS.get(name, name),
                    out_str,
                    _fmt_count(n),
                    _fmt_bytes(child_bytes),
                    f"{share:5.1f}%",
                )
            )
        _print_table(
            ["Step / module", "Output shape", "Params", "Size", "Share"],
            pipe_rows,
            aligns=["left", "left", "right", "right", "right"],
        )
        print(thin)
        print("  Module tree")
        print(thin)
        tree_rows = []
        self._collect_module_tree(
            self, prefix="", depth=0, max_depth=max_depth, total=total, rows=tree_rows
        )
        _print_table(
            ["Module", "Params", "Size", "Share"],
            tree_rows,
            aligns=["left", "right", "right", "right"],
        )

        print(thin)
        print("  Parameter tensors")
        print(thin)
        weight_rows = []
        for name, p in self.named_parameters():
            weight_rows.append(
                (
                    name,
                    _fmt_shape(p.shape),
                    _fmt_count(p.numel()),
                    _fmt_bytes(p.numel() * p.element_size()),
                )
            )
        _print_table(
            ["Tensor", "Shape", "Elements", "Size"],
            weight_rows,
            aligns=["left", "left", "right", "right"],
        )

        if dummy is not None and io_map is not None:
            print(thin)
            t1, t2, text = dummy
            print(
                "  Forward activation tensors  "
                f"(t1/t2 {_fmt_shape(t1.shape)},  text {_fmt_shape(text.shape)})"
            )
            print(thin)
            act_rows = [
                (
                    "input",
                    "t1_volume",
                    _fmt_shape(t1.shape),
                    _fmt_bytes(t1.numel() * t1.element_size()),
                ),
                (
                    "input",
                    "t2_volume",
                    _fmt_shape(t2.shape),
                    _fmt_bytes(t2.numel() * t2.element_size()),
                ),
                (
                    "input",
                    "text_embeddings",
                    _fmt_shape(text.shape),
                    _fmt_bytes(text.numel() * text.element_size()),
                ),
            ]
            self._collect_activation_rows(
                self, prefix="", depth=0, max_depth=max_depth, io_map=io_map, rows=act_rows
            )
            _print_table(
                ["Module", "Tensor", "Shape", "Size"],
                act_rows,
                aligns=["left", "left", "left", "right"],
            )

        print(bar)

    def _probe_forward(self, config):
        """Run a dummy forward pass and capture per-module input/output shapes."""
        p0 = next(self.parameters())
        patch = tuple(int(x) for x in config["data"]["patch"]["size"])
        batch = int(config["training"]["batch_size"])
        channels = int(config["model"]["input_channels"])
        n_prompts = len(config["data"]["prompts"])
        text_dim = int(config["model"]["text_dim"])

        def _make(batch_size):
            t1 = torch.zeros(batch_size, channels, *patch, device=p0.device, dtype=p0.dtype)
            t2 = torch.zeros(batch_size, channels, *patch, device=p0.device, dtype=p0.dtype)
            text = torch.zeros(batch_size, n_prompts, text_dim, device=p0.device, dtype=p0.dtype)
            return t1, t2, text

        io_map = {}

        def hook(mod, args, out):
            captured_in = args[0] if len(args) == 1 else args
            io_map[id(mod)] = (_capture(captured_in), _capture(out))

        handles = [mod.register_forward_hook(hook) for mod in self.modules()]
        was_training = self.training
        self.eval()
        dummy = None
        try:
            with torch.no_grad():
                try:
                    dummy = _make(batch)
                    self(*dummy)
                except RuntimeError as exc:
                    print(f"  Warning: dummy forward with batch={batch} failed ({exc}). Retrying batch=1.")
                    io_map.clear()
                    dummy = _make(1)
                    self(*dummy)
        except RuntimeError as exc:
            print(f"  Warning: could not collect activation shapes ({exc}).")
            io_map = None
            dummy = None
        finally:
            for h in handles:
                h.remove()
            self.train(was_training)
        return io_map, dummy

    def _collect_module_tree(self, module, prefix, depth, max_depth, total, rows):
        for name, child in module.named_children():
            n, _, _, nbytes = _numel_stats(child)
            if n == 0:
                continue
            share = (100.0 * n / total) if total else 0.0
            rows.append(
                (
                    f"{prefix}{name} ({type(child).__name__})",
                    _fmt_count(n),
                    _fmt_bytes(nbytes),
                    f"{share:5.1f}%",
                )
            )
            if depth + 1 < max_depth:
                self._collect_module_tree(
                    child, prefix + "  ", depth + 1, max_depth, total, rows
                )

    def _collect_activation_rows(self, module, prefix, depth, max_depth, io_map, rows):
        for name, child in module.named_children():
            n, _, _, _ = _numel_stats(child)
            if n == 0:
                continue
            label = f"{prefix}{name} ({type(child).__name__})"
            captured = io_map.get(id(child))
            if captured is not None:
                _, out_c = captured
                items = list(_iter_captured(out_c))
                labels = _output_labels(child, len(items))
                for i, (default_name, shape, nbytes) in enumerate(items):
                    tensor_name = labels[i] if labels is not None else default_name
                    rows.append(
                        (
                            label if i == 0 else "",
                            tensor_name,
                            _fmt_shape(shape),
                            _fmt_bytes(nbytes),
                        )
                    )
            else:
                rows.append((label, "", "", ""))
            if depth + 1 < max_depth:
                self._collect_activation_rows(
                    child, prefix + "  ", depth + 1, max_depth, io_map, rows
                )

    def forward(self, t1_volume, t2_volume, text_embeddings):
        # t1_volume:       Shape [B, 1, D_t1, H_t1, W_t1]
        # t2_volume:       Shape [B, 1, D_t2, H_t2, W_t2]  (may be unregistered)
        # text_embeddings: Shape [B, N_T, text_dim]

        # ==========================================
        # STEP 1: Feature Extraction (Encoders)
        # ==========================================
        # T1 path captures skip connections
        t1_bottleneck, skip3, skip2, skip1 = self.t1_encoder(t1_volume)
        # t1_bottleneck shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]

        # T2 path does not capture skip connections
        t2_bottleneck, _, _, _ = self.t2_encoder(t2_volume)
        # t2_bottleneck shape: [B, 128, D_t2//8, H_t2//8, W_t2//8]

        # ==========================================
        # STEP 2: Spatial Alignment (Cross-Volume MHA)
        # ==========================================
        # Projects Diffusion features to align with the T1 bottleneck layout
        fused_visual_map = self.cross_volume_attention(
            t1_features=t1_bottleneck,
            secondary_features=t2_bottleneck,
        )
        # fused_visual_map shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]

        # ==========================================
        # STEP 3: Semantic Alignment (Prompt Decoder MHA)
        # ==========================================
        # Aligns text tokens to the unified spatial visual feature map
        aligned_queries = self.prompt_decoder(
            text_embeddings=text_embeddings, 
            fused_visual_features=fused_visual_map
        ) # Shape: [B, N_T, 128]

        # ==========================================
        # STEP 4: Reconstruction (Multi-Scale Decoder)
        # ==========================================
        skips = [skip1, skip2, skip3]
        
        # Decoder performs channel-wise scaling and outputs multi-resolution masks
        stage_predictions = self.decoder(
            fused_bottleneck=fused_visual_map,
            skips=skips,
            aligned_queries=aligned_queries,
        )
        
        # Returns list of 3D mask logits for deep supervision loss computation:
        # stage_predictions[0]: Lowest resolution [B, N_T, D_t1//4, H_t1//4, W_t1//4]
        # stage_predictions[1]: Middle resolution [B, N_T, D_t1//2, H_t1//2, W_t1//2]
        # stage_predictions[2]: Final resolution  [B, N_T, D_t1, H_t1, W_t1]
        return stage_predictions
