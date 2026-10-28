"""Fixed enabling configuration; experimental tuning and model math are removed."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Config:
    name: str = 'thor-native-port'
    workers: int = 20

    def validate(self, device):
        if type(self.workers) is not int or not 1 <= self.workers <= device['sm_count']:
            return ['workers must fit physical SM residency']
        if (device.get('cc_major'), device.get('cc_minor')) != (11, 0):
            return ['requires CC 11.0']
        return []

    def to_dict(self):
        return asdict(self)
