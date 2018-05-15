import argparse
from . import pick_and_place


def main():
    parser = argparse.ArgumentParser(description="Entry point to a collection of programs for object"
                                                 "transportation I wrote while working on a paper.")
    subparsers = parser.add_subparsers()
    # A subparser for the program that computes the robust
    # controllable sets
    parser_pick = subparsers.add_parser('pick-demo', description='')
    parser_pick.set_defaults(which='pick-demo')
    parser_pick.add_argument('-s', "--scene", help="Path to scenario. "
                                                   "Example: scenarios/test0.scenario.yaml",
                             default="scenarios/test0.scenario.yaml")
    parser_pick.add_argument('-v', dest='verbose', action='store_true')
    parser_pick.add_argument('-e', "--execute_hw", help="If True, send commands to real hardware.",
                             action="store_true", default=False)
    args = parser.parse_args()
    if args.which == "pick-demo":
        pick_and_place.main(load_path=args.scene, verbose=args.verbose,
                            execute_hw=args.execute_hw)


    return 1
