import os
import signal

os.kill(int(os.getenv("PID", None)), signal.SIGUSR1)
