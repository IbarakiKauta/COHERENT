import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curr_dir, "..", ".."))

from utils.experiment_config import build_common_parser


def get_args():
    parser = build_common_parser(include_rounds=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()