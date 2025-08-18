import psutil
import time
from functools import wraps
import os

"""
Simple profiler for individual functions

Annotation: 

- In igua modules, import as `from .profiler import profiler`. 
- Then, use the `@profiler.profile_function` decorator to profile individual functions.

Usage: 

# profiling quiet
IGUA_PROFILE=1 python -m igua -i strains_metadata.tsv --output advanced_profile/gcfs_prof.tsv

# profiling verbose
IGUA_PROFILE=1 IGUA_VERBOSE=1 python -m igua -i strains_metadata.tsv --output advanced_profile/gcfs_prof.tsv

"""

class MemoryProfiler:
    def __init__(self):
        self.enabled = os.getenv('IGUA_PROFILE', '0') == '1'
        self.function_stats = {}
        
    def profile_function(self, func):
        """Decorator to profile memory usage of functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
                
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                mem_after = process.memory_info().rss / 1024 / 1024
                duration = time.time() - start_time
                mem_delta = mem_after - mem_before
                
                func_name = f"{func.__module__}.{func.__name__}"
                if func_name not in self.function_stats:
                    self.function_stats[func_name] = []
                    
                self.function_stats[func_name].append({
                    'memory_delta': mem_delta,
                    'duration': duration,
                    'memory_after': mem_after
                })
                
                if os.getenv('IGUA_VERBOSE', '0') == '1':
                    print(f"{func_name}: {mem_delta:+.1f} MB in {duration:.1f}s")
                    
                return result
                
            except Exception as e:
                print(f"{func.__module__}.{func.__name__} failed: {e}")
                raise
                
        return wrapper
    
    def report(self):
        """Print memory usage summary"""
        if not self.enabled or not self.function_stats:
            return
            
        print("\n" + "="*50)
        print("MEMORY USAGE REPORT")
        print("="*50)
        
        for func_name, calls in self.function_stats.items():
            if not calls:
                continue
                
            total_memory = sum(call['memory_delta'] for call in calls if call['memory_delta'] > 0)
            avg_memory = sum(call['memory_delta'] for call in calls) / len(calls)
            max_memory = max(call['memory_delta'] for call in calls)
            total_time = sum(call['duration'] for call in calls)
            
            print(f"{func_name}:")
            print(f"  Calls: {len(calls)}")
            print(f"  Memory: {avg_memory:+.1f} MB avg, {max_memory:+.1f} MB max")
            print(f"  Time: {total_time:.1f}s total")
            print()

profiler = MemoryProfiler()
