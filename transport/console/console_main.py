import argparse


def main():
    parser = argparse.ArgumentParser(description="Entry point to a collection of programs for object transportation..")
    subparsers = parser.add_subparsers()
    # A subparser for the program that computes the robust
    # controllable sets
    parser_robust = subparsers.add_parser('', description='')
    parser_robust.set_defaults(which='robust_sets')
    parser_robust.add_argument('-v', action='store_true', dest='verbose', help='On for verbose')
    parser_robust.add_argument('-s', "--savefig", action='store_true', help='If true save the figure.', default=False)
   
