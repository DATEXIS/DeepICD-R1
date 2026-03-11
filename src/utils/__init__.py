import functools
import time


def log_function_call(logger):
    def log_function_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)

            logger.info(f"Calling {func.__name__}({signature})")
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()
            minutes, seconds = divmod(end_time - start_time, 60)

            logger.info(
                f"Finished {func.__name__} in {int(minutes)}m {seconds:.2f}s. "
                f"Results: {repr(result)}"
            )
            return result

        return wrapper

    return log_function_call_decorator
