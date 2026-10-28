from dataclasses import dataclass, fields
from typing import Any

from torch import Tensor

from megakernels.model_types import DeviceType
from megakernels.utils import get_sm_count


@dataclass
class BaseGlobals:
    # model parameters, all layers stacked together in order
    qkv_proj_weights: Tensor
    attn_ln_weights: Tensor
    o_proj_weights: Tensor
    mlp_ln_weights: Tensor
    up_proj_weights: Tensor
    gate_proj_weights: Tensor
    down_proj_weights: Tensor
    lm_head_norm_weights: Tensor
    lm_head_weights: Tensor
    k_cache: Tensor
    v_cache: Tensor

    # not stacked for each layer
    rope_cos: Tensor
    rope_sin: Tensor

    # model constants
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    attn_scale: float
    rms_norm_eps: float
    device: DeviceType

    hidden_states: Tensor
    barriers: Tensor

    pos_id: int

    def __post_init__(self):
        self.instructions: Tensor | None = None
        self.timings: Tensor | None = None

    def physical_sm_count(self) -> int:
        return get_sm_count(self.device)

    def sm_count(self) -> int:
        physical = self.physical_sm_count()
        active = getattr(self, "active_workers", None)
        if active is None:
            return physical
        limit = getattr(self, "max_ctas_per_sm", 1)
        if type(limit) is not int or limit not in (1, 2):
            raise ValueError("max_ctas_per_sm must be 1 or 2")
        if not isinstance(active, int) or isinstance(active, bool) or not 1 <= active <= physical * limit:
            raise ValueError("active_workers exceeds the explicit physical-SM residency budget")
        return active

    def set_active_workers(self, count: int, *, max_ctas_per_sm: int = 1):
        if type(count) is not int:
            raise ValueError("active worker count must be an integer")
        previous = getattr(self, "active_workers", None)
        previous_limit = getattr(self, "max_ctas_per_sm", 1)
        if type(max_ctas_per_sm) is not int or max_ctas_per_sm not in (1, 2):
            raise ValueError("max_ctas_per_sm must be 1 or 2")
        self.active_workers = count
        self.max_ctas_per_sm = max_ctas_per_sm
        try:
            self.sm_count()
        except ValueError:
            self.active_workers = previous
            self.max_ctas_per_sm = previous_limit
            raise
        # Existing queue tensors were built for the previous worker count.
        self.instructions = self.timings = None

    def diff(self, other: "BaseGlobals", skip_kv_cache: bool = False):
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)
            other_attr = getattr(other, name)
            if (
                not isinstance(attr, Tensor)
                or "weights" in name
                or name in ["rope_cos", "rope_sin"]
                or (skip_kv_cache and "cache" in name)
            ):
                continue
            diff_tensors(attr, other_attr, name)

    def num_total_heads(self) -> int:
        return self.num_attention_heads + self.num_kv_heads * 2


def diff_tensors(a: Tensor, b: Tensor, name: str):
    a = a.float()
    b = b.float()

    diff = a - b
    adiff = diff.abs()
    rdiff = 2 * adiff / (a.abs() + b.abs() + 1e-6)
    print(f"{name}: max adiff: {adiff.max()}, mean rdiff: {rdiff.mean()}")
    return diff, adiff, rdiff


@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def prev_opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def tags(cls) -> dict[str, Any]:
        return {}

    def serialize(self):
        words = [self.opcode()]
        for field in fields(self):
            name = field.name
            if name == "global_idx":
                continue
            attr = getattr(self, name)

            if isinstance(attr, int):
                words.append(attr)
            elif isinstance(attr, tuple):
                words.append(len(attr))
                words.extend(attr)
            elif isinstance(attr, list):
                words.append(len(attr))
                words.extend(attr)
            # for convenience
            elif attr is None:
                words.append(0)
            else:
                raise ValueError(f"Unsupported field type: {attr}")

        return words


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0


@dataclass
class PrintInfo:
    layer_filter: list[int] | None = None
    name_filter: list[str] | None = None
    state_filter: list[str] | None = None


@dataclass
class PrintState(Instruction):
    layer_idx: int
    name: str
    print_info: PrintInfo
