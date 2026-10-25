"""Exercise real subprocess pipes and parent-death cleanup on the Linux runner."""
import os
import pathlib
import subprocess
import sys
import tempfile
import time

wrapper = pathlib.Path(__file__).with_name('hdfs-maven.py')
with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, HDFS_REAL_MVN=sys.executable, HDFS_MAVEN_STDERR_LOG=str(pathlib.Path(directory) / 'maven.stderr'))
    result = subprocess.run([sys.executable, str(wrapper), '-u', '-c', "import sys; print('startup diagnostic'); print('Ready!'); sys.stdin.readline(); print('finished')"], input=b'\n', capture_output=True, env=env, timeout=10)
    assert result.returncode == 0 and result.stdout == b'Ready!\nfinished\n'
    assert (pathlib.Path(directory) / 'maven.stdout-before-ready').read_bytes() == b'startup diagnostic\n'
    missing = subprocess.run([sys.executable, str(wrapper), '-u', '-c', "print('startup failed')"], capture_output=True, env=env, timeout=10)
    assert missing.returncode != 0 and missing.stdout == b''
    child = subprocess.Popen([sys.executable, str(wrapper), '-u', '-c', "import time; print('Ready!'); time.sleep(60)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        assert child.stdout.readline() == b'Ready!\n'
        real_pid = int((pathlib.Path(directory) / 'maven-child-pid').read_text())
        proc = pathlib.Path('/proc') / str(real_pid) / 'stat'
        started = proc.read_text().rsplit(')', 1)[1].split()[19]
        child.kill()
        child.communicate(timeout=5)
        deadline = time.monotonic() + 5
        while proc.exists():
            try:
                state = proc.read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                break
            if state[0] == 'Z' or state[19] != started:
                break
            assert time.monotonic() < deadline, 'real child survived wrapper death'
            time.sleep(0.05)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
print('Readiness forwarding, missing-readiness rejection, stdin, and parent-death cleanup pass.')
