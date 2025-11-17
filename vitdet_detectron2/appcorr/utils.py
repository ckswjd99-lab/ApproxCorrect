import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
import functools

from typing import Tuple

def register_cache(fname: str, cache: Dict[str, torch.Tensor], tensor: torch.Tensor):
    cache[fname] = tensor.clone()

    return tensor

def timer_start():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event, end_event

def timer_end(start_event, end_event, description="Elapsed time"):
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    # print(f"{description}: {elapsed_time} ms")
    return elapsed_time

def cuda_timer(accumulator_attr=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, 'debug_time', False):
                return func(self, *args, **kwargs)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = func(self, *args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)

            if accumulator_attr:
                current_total = getattr(self, accumulator_attr, 0)
                setattr(self, accumulator_attr, current_total + elapsed_time)
                print(f"[Timer] {func.__name__}: {elapsed_time} ms (Accumulated: {getattr(self, accumulator_attr)} ms)")

            return result
        return wrapper
    return decorator
