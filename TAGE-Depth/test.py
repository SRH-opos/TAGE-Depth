"""Quick sanity check."""

import subprocess
import sys


def main():
    subprocess.check_call([sys.executable, 'sample_inference.py'])
    print('TAGE-Depth quick test passed.')


if __name__ == '__main__':
    main()
