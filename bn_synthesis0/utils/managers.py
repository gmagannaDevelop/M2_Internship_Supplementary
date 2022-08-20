"""
    The interface to get tidy CSVs from messy CSVs.
"""

from multiprocessing import Process


class TimedProcess:
    """Wrapper for multiprocessing.Process"""

    def __init__(
        self,
        timeout: int,
        group=None,
        target=None,
        name=None,
        daemon=None,
        *args,
        **kwargs,
    ):
        self.timeout = timeout
        self.group = group
        self.target = target
        self.name = name
        self.daemon = daemon
        self.process_args = args
        self.process_kwargs = kwargs
        self._process: Process  # declared but instantiated in __enter__

    def __repr__(self):
        return f"TimedProcess context manager at {hex(id(self))}"

    def __enter__(self):
        self._process = Process(
            group=self.group,
            target=self.target,
            name=self.name,
            args=self.process_args,
            kwargs=self.process_kwargs,
        )
        self._process.start()
        self._process.join(timeout=self.timeout)
        return self._process

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._process.terminate()
