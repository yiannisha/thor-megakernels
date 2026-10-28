"""Constraints for the currently implemented Thor layout; no speculative stages."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    name: str = 'upstream-like'
    consumer_warps: int = 16
    consumer_registers: int = 104
    service_registers: int = 64
    register_mode: str = 'static'
    static_registers: int = 96
    workers: int = 20
    backoff_ns: int = 20
    scheduler: str = 'rr'
    instruction_parties: int = 32
    ordered_downproj: bool = False
    fp32_downproj: bool = False
    match_torch_bf16_epilogues: bool = False
    compensated_attention: bool = False
    fp64_rms: bool = False
    tensorcore_matvec: bool = False
    match_torch_bf16_rms: bool = True
    timing: bool = False
    allow_multi_cta: bool = False
    ready_queue: bool = False
    serial_instruction_handoff: bool = False
    arch: str = 'sm_110a'
    page_size: int = 16384
    page_count: int = 13
    weight_stages: int = 3
    tile_width: int = 512
    architecture: str = 'static'
    precision: str = 'bf16'

    def validate(self, device):
        errors = []
        if any(type(value) is not bool for value in (self.timing, self.allow_multi_cta, self.ready_queue, self.serial_instruction_handoff, self.match_torch_bf16_rms, self.ordered_downproj, self.fp32_downproj, self.match_torch_bf16_epilogues, self.compensated_attention, self.fp64_rms, self.tensorcore_matvec)):
            errors.append('configuration switches must be booleans')
        if self.ready_queue and (not self.serial_instruction_handoff or self.allow_multi_cta or self.timing or self.register_mode!='static'):
            errors.append('ready prototype requires static, serial, untimed single-CTA residency')
        if self.ready_queue and self.scheduler!='rr':
            errors.append('ready dispatch uses device FIFO; static scheduler overrides are unsupported')
        if self.tensorcore_matvec and not self.match_torch_bf16_rms:
            errors.append('BF16 MMA matvec requires BF16-rounded normalized inputs')
        if self.fp32_downproj and self.ordered_downproj:
            errors.append('FP32 downprojection and ordered BF16 updates are mutually exclusive')
        integer_fields = ('instruction_parties', 'static_registers', 'consumer_warps', 'consumer_registers', 'service_registers', 'workers',
                          'backoff_ns', 'page_size', 'page_count', 'weight_stages', 'tile_width')
        if any(type(getattr(self, field)) is not int for field in integer_fields):
            return ['configuration dimensions must be integers']
        if self.instruction_parties != 32:
            errors.append('Thor requires per-thread instruction barriers; warp-aggregated path failed racecheck')
        if self.consumer_warps not in (4, 8, 16):
            errors.append('consumer warps must be 4, 8 or 16')
        for registers in (self.consumer_registers, self.service_registers):
            if registers < 24 or registers > 256 or registers % 8:
                errors.append('setmaxnreg requires multiples of 8 in 24..256')
        if self.register_mode == 'dynamic' and self.arch != 'sm_110a':
            errors.append('dynamic registers require architecture-specific sm_110a codegen')
        if self.register_mode not in ('static', 'dynamic'):
            errors.append('unknown register allocation mode')
        if self.register_mode == 'static' and (self.consumer_registers, self.service_registers) != (104, 64):
            errors.append('separate consumer/service budgets require dynamic register mode')
        if self.register_mode == 'dynamic' and 32 * (self.consumer_warps * self.consumer_registers + 4 * self.service_registers) > 65536:
            errors.append('dynamic register budgets exceed per-SM register pool')
        if not 24 <= self.static_registers <= 255 or self.static_registers % 8:
            errors.append('static register cap must be a multiple of 8 in 24..248')
        # A compiler cap is an upper bound, not a fixed per-thread allocation.
        # Reject actual compiled occupancy at native preflight.
        if self.allow_multi_cta and self.register_mode != 'static':
            errors.append('multi-CTA prototype currently requires static register allocation')
        if not 1 <= self.workers <= device['sm_count'] * (2 if self.allow_multi_cta else 1):
            errors.append('workers exceed explicitly enabled residency budget or are zero')
        stage_pages = 2048 // self.tile_width if self.tile_width in (256, 512) else 0
        max_pages = 25 if self.page_size == 8192 else 13
        if self.weight_stages not in (1, 2, 3) or not 1 + stage_pages * self.weight_stages <= self.page_count <= max_pages:
            errors.append('weight stages require 1..3 stages and sufficient activation/weight pages')
        if (self.page_size, self.tile_width) not in ((16384, 512), (8192, 256)):
            errors.append('supported page/chunk pairs are 16KiB/512 and 8KiB/256')
        if stage_pages and self.consumer_warps % stage_pages:
            errors.append('consumer warps must divide evenly across weight-stage pages')
        if self.scheduler not in ('rr', 'zz', 'wave', 'dag'):
            errors.append('unsupported static scheduler')
        if self.arch not in ('sm_110', 'sm_110a'):
            errors.append('unsupported Thor compilation target')
        if self.architecture != 'static' or self.precision != 'bf16':
            errors.append('architecture/precision prototype not implemented')
        if not 0 <= self.backoff_ns <= 1000000:
            errors.append('invalid nanosleep backoff')
        if (device.get('cc_major'), device.get('cc_minor')) != (11, 0):
            errors.append('requires CC 11.0')
        return errors

    def defines(self):
        return [f'-D{key}={value}' for key, value in {
            'MK_WEIGHT_PIPELINE_STAGES': self.weight_stages,
            'MK_PAGE_COUNT': self.page_count,
            'MK_PAGE_SIZE': self.page_size,
            'MK_WEIGHT_CHUNK_WIDTH': self.tile_width,
            'MK_THOR_DYNAMIC_REGISTERS': int(self.register_mode == 'dynamic'),
            'MK_CONSUMER_WARPS': self.consumer_warps,
            'MK_CONSUMER_REGISTERS': self.consumer_registers,
            'MK_SERVICE_REGISTERS': self.service_registers,
            'MK_BACKOFF_NS': self.backoff_ns,
            'MK_RECORD_TIMING': int(self.timing),
            'MK_ALLOW_MULTI_CTA': int(self.allow_multi_cta),
            'MK_READY_QUEUE': int(self.ready_queue),
            'MK_SERIAL_INSTRUCTION_HANDOFF': int(self.serial_instruction_handoff),
            'MK_MATCH_TORCH_BF16_RMS': int(self.match_torch_bf16_rms),
            'MK_ORDERED_DOWNPROJ': int(self.ordered_downproj),
            'MK_FP32_DOWNPROJ': int(self.fp32_downproj),
            'MK_FP64_RMS': int(self.fp64_rms),
            'MK_TENSORCORE_MATVEC': int(self.tensorcore_matvec),
            'MK_COMPENSATED_ATTENTION': int(self.compensated_attention),
            'MK_MATCH_TORCH_BF16_EPILOGUES': int(self.match_torch_bf16_epilogues),
            'MK_INSTRUCTION_BARRIER_PARTIES': self.instruction_parties}.items()]

    def to_dict(self):
        return asdict(self)
