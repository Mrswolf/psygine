__all__ = ["profile"]

from line_profiler import LineProfiler


def profile(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        result = profiler.runcall(func, *args, **kwargs)
        profiler.print_stats(output_unit=1e-3)
        return result

    return wrapper
