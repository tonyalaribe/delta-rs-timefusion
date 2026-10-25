"""Keep MiniDfs stdout for its readiness protocol and supervise the real JVM."""
import ctypes
import os
import pathlib
import signal
import subprocess
import sys


def main():
    diagnostics = pathlib.Path(os.environ['HDFS_MAVEN_STDERR_LOG']).parent
    parent = os.getpid()
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    prctl.restype = ctypes.c_int

    def supervise_child():
        # Linux keeps this across exec, so MiniDfs.kill() also kills the real JVM.
        if prctl(1, signal.SIGKILL, 0, 0, 0):  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'cannot supervise Maven')
        if os.getppid() != parent:
            os._exit(1)

    ready = False
    prelude_bytes = 0
    with (diagnostics / 'maven.stderr').open('ab') as stderr, (diagnostics / 'maven.stdout-before-ready').open('ab') as prelude:
        child = subprocess.Popen([os.environ['HDFS_REAL_MVN'], *sys.argv[1:]], stdout=subprocess.PIPE, stderr=stderr, preexec_fn=supervise_child)
        (diagnostics / 'maven-child-pid').write_text(str(child.pid))
        try:
            for line in child.stdout:
                if not ready and line.rstrip(b'\r\n') == b'Ready!':
                    ready = True
                if ready:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                else:
                    prelude.write(line)
                    prelude.flush()
                    prelude_bytes += len(line)
                    if prelude_bytes > 1 << 20:
                        raise RuntimeError('MiniDfs startup preamble exceeded 1 MiB')
            code = child.wait()
            return code if ready else (code or 1)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == '__main__':
    raise SystemExit(main())
