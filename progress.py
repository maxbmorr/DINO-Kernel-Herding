import sys
from time import perf_counter


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(
        self, total, description="Progress", initial=0, width=30,
        refresh_interval=1.0,
    ):
        self.total = max(0, int(total))
        self.description = description
        self.current = max(0, int(initial))
        self.width = width
        self.refresh_interval = float(refresh_interval)
        self.started = perf_counter()
        self._last_rendered_at = 0.0
        self._render()

    def update(self, amount=1, detail=None):
        self.current = min(self.total, self.current + amount)
        now = perf_counter()
        if (
            self.current == self.total
            or now - self._last_rendered_at >= self.refresh_interval
        ):
            self._render(detail)

    def _render(self, detail=None):
        ratio = self.current / self.total if self.total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = perf_counter() - self.started
        rate = (self.current / elapsed) if elapsed > 0 else 0.0
        eta = ((self.total - self.current) / rate) if rate > 0 else 0.0
        suffix = f" | {detail}" if detail else ""
        sys.stdout.write(
            f"\r{self.description} [{bar}] {self.current}/{self.total} "
            f"({ratio:6.1%}) ETA {eta:6.1f}s{suffix}"
        )
        sys.stdout.flush()
        self._last_rendered_at = perf_counter()

    def close(self):
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()
