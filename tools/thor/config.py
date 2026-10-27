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
    match_torch_bf16_rms: bool = True
    timing: bool = False
    arch: str = 'sm_110a'
    page_size: int = 16384
    page_count: int = 13
    weight_stages: int = 3
    tile_width: int = 512
    architecture: str = 'static'
    precision: str = 'bf16'

    def validate(self, device):
        errors = []
        if any(type(value) is not bool for value in (self.timing, self.match_torch_bf16_rms, self.ordered_downproj, self.fp32_downproj, self.match_torch_bf16_epilogues)):
            errors.append('timing and RMS rounding switches must be booleans')
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
        # launch_bounds may lower it; reject actual occupancy at native preflight.
        if not 1 <= self.workers <= device['sm_count']:
            errors.append('workers exceed physical SM count or are zero')
        if self.weight_stages not in (1, 2, 3) or not 1 + 4 * self.weight_stages <= self.page_count <= 13:
            errors.append('weight stages require 1..3 stages and at least one activation plus four pages per stage')
        if (self.page_size, self.tile_width) != (16384, 512):
            errors.append('only 16KiB pages and 512-wide weight chunks currently supported')
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
            'MK_THOR_DYNAMIC_REGISTERS': int(self.register_mode == 'dynamic'),
            'MK_CONSUMER_WARPS': self.consumer_warps,
            'MK_CONSUMER_REGISTERS': self.consumer_registers,
            'MK_SERVICE_REGISTERS': self.service_registers,
            'MK_BACKOFF_NS': self.backoff_ns,
            'MK_RECORD_TIMING': int(self.timing),
            'MK_MATCH_TORCH_BF16_RMS': int(self.match_torch_bf16_rms),
            'MK_ORDERED_DOWNPROJ': int(self.ordered_downproj),
            'MK_FP32_DOWNPROJ': int(self.fp32_downproj),
            'MK_MATCH_TORCH_BF16_EPILOGUES': int(self.match_torch_bf16_epilogues),
            'MK_INSTRUCTION_BARRIER_PARTIES': self.instruction_parties}.items()]

    def to_dict(self):
        return asdict(self)
