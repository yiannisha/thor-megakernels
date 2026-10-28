import torch
from einops import rearrange

from megakernels.demos.latency.instructions import Globals
from megakernels.mk import MK_Interpreter


def interpret_with_mk(
    globs: Globals,
    mk_func,
):
    if globs.ready_metadata is None:
        globs.ready_metadata = torch.zeros(1,dtype=torch.int32,device=globs.device)
        globs.ready_state = torch.zeros(1,dtype=torch.int32,device=globs.device)
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # vm stuff
        globs.barriers,
        globs.instructions,
        globs.timings,
        globs.ready_metadata, globs.ready_state,
        # weights
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,
        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        globs.down_partials,
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,
        stream=torch.cuda.current_stream(),
    )


class LatencyMK_Interpreter(MK_Interpreter):
    def interpret(self, globs: Globals):
        interpret_with_mk(globs, self.mk_func)
