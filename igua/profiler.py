import psutil
import time
from functools import wraps
import os
import rich.console
import rich.table

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

# command line -- quiet 
igua -i strains_metadata.tsv --write-defense-systems defsysts --output gcfs_prof.tsv --profile-memory 
# or 
igua -i strains_metadata.tsv --write-defense-systems defsysts --output gcfs_prof.tsv --profile-memory quiet

# command line -- verbose
igua -i strains_metadata.tsv --write-defense-systems defsysts --output gcfs_prof.tsv --profile-memory verbose
"""

HIGH_MEMORY_MB = 100
MED_MEMORY_MB = 20

class MemoryProfiler:
    def __init__(self):
        self.enabled = os.getenv('IGUA_PROFILE', '0') == '1'
        self.function_stats = {}
        self.console = rich.console.Console()
        
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
                    if mem_delta > 10:
                        color = "red"
                    elif mem_delta > 1:
                        color = "yellow"
                    elif mem_delta < -1:
                        color = "green"
                    else:
                        color = "blue"
                    
                    self.console.print(
                        f"\n[bold dark_orange]{'!! Profile':>12}[/][{color}] {func_name}: " #func_name.split('.')[-1]
                        f"[bold {color}]{mem_delta:+.1f} MB[/] [dim white]in {duration:.1f}s[/]"
                    )
                    
                return result
                
            except Exception as e:
                self.console.print(f"[bold red]{'Error':>12}[/] {func.__module__}.{func.__name__} failed: {e}")
                raise
                
        return wrapper
    
    def report(self):
        """Print memory usage summary using rich formatting"""
        if not self.enabled or not self.function_stats:
            return
            
        total_functions = len(self.function_stats)
        total_calls = sum(len(calls) for calls in self.function_stats.values())
        total_memory_increase = sum(
            sum(call['memory_delta'] for call in calls if call['memory_delta'] > 0)
            for calls in self.function_stats.values()
        )
        total_time = sum(
            sum(call['duration'] for call in calls)
            for calls in self.function_stats.values()
        )
        
        self.console.print("\n" * 2) 
        self.console.rule("[bold dark_orange]Memory Usage Report", style="dark_orange")
        
        summary_table = rich.table.Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_row("[bold white]Functions profiled:", f"[bold green]{total_functions}")
        summary_table.add_row("[bold white]Total function calls:", f"[bold green]{total_calls}")
        summary_table.add_row("[bold white]Total memory increase:", f"[bold red]{total_memory_increase:.1f} MB")
        summary_table.add_row("[bold white]Total execution time:", f"[bold yellow]{total_time:.1f} seconds")

        self.console.print(summary_table)
        
        # details table
        table = rich.table.Table(show_header=True, header_style="bold blue")
        table.add_column("Function", style="cyan", no_wrap=True)
        table.add_column("Calls", justify="right", style="green")
        table.add_column("Avg Memory", justify="right", style="yellow")
        table.add_column("Max Memory", justify="right", style="red")
        table.add_column("Total Time", justify="right", style="magenta")
        table.add_column("Status", justify="center")
        
        # sort by total memory impact
        sorted_functions = sorted(
            self.function_stats.items(),
            key=lambda x: sum(call['memory_delta'] for call in x[1] if call['memory_delta'] > 0),
            reverse=True
        )
        
        for func_name, calls in sorted_functions:
            if not calls:
                continue
                
            avg_memory = sum(call['memory_delta'] for call in calls) / len(calls)
            max_memory = max(call['memory_delta'] for call in calls)
            total_time = sum(call['duration'] for call in calls)
            
            # status indicator based on memory usage
            if max_memory > HIGH_MEMORY_MB:
                status = "[bold red]HIGH"
            elif max_memory > MED_MEMORY_MB:
                status = "[bold yellow]MED"
            elif avg_memory < 0:
                status = "[bold green]CLEAN"
            else:
                status = "[bold blue]OK"
            
            table.add_row(
                func_name,
                str(len(calls)),
                f"{avg_memory:+.1f} MB",
                f"{max_memory:+.1f} MB", 
                f"{total_time:.1f}s",
                status
            )
        
        self.console.print(table)
        
        high_memory_functions = [
            func_name for func_name, calls in self.function_stats.items()
            if calls and max(call['memory_delta'] for call in calls) > HIGH_MEMORY_MB
        ]
        
        if high_memory_functions:
            self.console.print() 
            self.console.print("[bold red]High Memory Usage Functions:")
            for func_name in high_memory_functions:
                self.console.print(f"   [red]{func_name}")
        
        self.console.rule(style="dark_orange")

profiler = MemoryProfiler()