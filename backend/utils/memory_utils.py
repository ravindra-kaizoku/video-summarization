import os

def is_render_environment() -> bool:
    """Returns True if running on Render cloud hosting."""
    return (
        os.environ.get("RENDER") == "true" or
        os.environ.get("RENDER_SERVICE_ID") is not None or
        os.environ.get("IS_RENDER") == "true"
    )

def get_available_memory_mb() -> float:
    """Returns available system memory in MB. On Linux/Render, reads /proc/meminfo."""
    try:
        # On Render's 512MB free container, always enforce conservative memory limit
        if is_render_environment():
            # Check cgroup memory limits if available
            cgroup_limit_paths = [
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory.max"
            ]
            for p in cgroup_limit_paths:
                if os.path.exists(p):
                    try:
                        with open(p, "r") as f:
                            val = int(f.read().strip())
                            if val < 10**12:  # not unlimited
                                return 150.0  # Under 512MB cgroup, force lightweight safe mode
                    except Exception:
                        pass
            return 180.0  # Default conservative limit for Render free tier

        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 1024.0
