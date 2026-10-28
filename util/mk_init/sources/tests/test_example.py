from {{PROJECT_NAME_LOWER}} import example_megakernel
import torch

workers = torch.cuda.get_device_properties(0).multi_processor_count
instruction = torch.zeros(workers, 1, 32, dtype=torch.int32, device="cuda")
timing = torch.zeros(workers, 1, 128, dtype=torch.int32, device="cuda")
instruction[0, 0, 0] = 1

print('Starting example megakernel')
example_megakernel(instruction, timing)