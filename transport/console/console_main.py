import argparse
from . import pick_and_place
from . import simplify_contact
from . import robust_experiment
import yaml
import numpy as np

DEFAULT_SCENE_PATH = "/home/hung/git/toppra-object-transport/models/robust-exp.env.xml"
DEFAULT_ROBOT_NAME = "denso"


def main():
    parser = argparse.ArgumentParser(description="Entry point to a collection of programs for object "
                                                 "transportation I wrote while working on a paper.")
    subparsers = parser.add_subparsers()
    ###########################################################################
    #                        Pick-and-place subparser demo                    #
    ###########################################################################
    parser_pick = subparsers.add_parser('pick-demo', description='')
    parser_pick.set_defaults(which='pick-demo')  # use this to select sub-parser
    parser_pick.add_argument('-s', "--scene", help="Path to the scenario file. "
                                                   "Example: scenarios/test0.scenario.yaml",
                             default="scenarios/test0.scenario.yaml")
    parser_pick.add_argument('-v', dest='verbose', action='store_true')
    parser_pick.add_argument('-d', "--slowdown", type=float, default=0.5)
    parser_pick.add_argument('-e', "--execute_hw", help="If True, send commands to real hardware.",
                             action="store_true", default=False)

    ###########################################################################
    #                        Simplify contact subparser demo                  #
    ###########################################################################
    parser_sim = subparsers.add_parser('simplify-contact', description="A program for simplifying and converting contact configurations. "
                                                                       "Contact should contain raw_data field.")
    parser_sim.set_defaults(which='simplify-contact')  # use this to select sub-parser
    parser_sim.add_argument('-c', '--contact', help='Profile id of the contact to be simplified', required=True)
    parser_sim.add_argument('-o', '--object', help='Profile id of the object specification, used for dynamic exploration.', required=False)
    parser_sim.add_argument('-a', '--attach', help='Name of the link or mnaipulator that the object is attached to.', required=False, default="denso_suction_cup")
    parser_sim.add_argument('-T', '--transform', help='Transformation T_link_object', required=False, default="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 9.080e-3], [0, 0, 0, 1]]")
    parser_sim.add_argument('-r', '--robot', help='Robot specification, use for dynamic exploration (strategy10).', required=False, default="suctioncup1")
    parser_sim.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    parser_sim.add_argument('-e', '--cover_vertices', help='If is true, cover extreme vertices', action="store_true")

    ###########################################################################
    #                              Robust experiment                          #
    ###########################################################################
    parser_rob = subparsers.add_parser('robust-experiment',
                                       description="A program for parametrizing and "
                                       "executing a single trajectory.")
    parser_rob.set_defaults(which='robust-experiment')
    parser_rob.add_argument("-s", '--scene_path', help="Path to an OpenRAVE scene.", default=DEFAULT_SCENE_PATH)
    parser_rob.add_argument("-r", '--robot_name', help="Name of the OpenRAVE robot.", default=DEFAULT_ROBOT_NAME)

    parser_rob.add_argument("-c", '--contact_id', help="Id of the contact profile", required=True)
    parser_rob.add_argument("-o", '--object_id', help="Id of the object profile", required=True)
    parser_rob.add_argument("-a", '--attach', help="Name of the link or manipulator the object is attached to.", required=True)
    parser_rob.add_argument("-T", '--transform', help="Transformation T_{link}_{object}", required=True)
    parser_rob.add_argument("-t", '--trajectory_id', help="Id of the trajectory to execute.", required=True)

    parser_rob.add_argument("-S", '--strategy', help="Parameterization strategy. "
                            "Can be `kin_only` or `w_contact`.", required=True)
    parser_rob.add_argument('-v', '--verbose', help="Show additional messages.", action='store_true')
    parser_rob.add_argument('-d', "--slowdown", type=float, default=1.0)
    parser_rob.add_argument('-e', "--execute", help="If True, send commands to real hardware. "
                            "Otherwise, only run in OpenRAVE environment.",
                            action="store_true", default=False)
    parser_rob.add_argument('-y', "--safety", type=float, default=1.0)  # DEPRECATED
    ###########################################################################
    #                           Run approprate programs                       #
    ###########################################################################

    args = parser.parse_args()
    if args.which == "pick-demo":
        pick_and_place.main(load_path=args.scene, verbose=args.verbose, execute_hw=args.execute_hw, slowdown=args.slowdown)
    elif args.which == "simplify-contact":
        simplify_contact.main(
            contact_id=args.contact,
            object_id=args.object,
            attach_name=args.attach,
            T_link_object=args.transform,
            robot_id=args.robot,
            verbose=args.verbose,
            cover_vertices=args.cover_vertices
        )
    elif args.which == 'robust-experiment':
        transform = np.array(yaml.load(args.transform), dtype=float)
        robust_experiment.main(None, args.scene_path,
                               args.robot_name, args.contact_id, args.object_id,
                               args.attach, transform, args.trajectory_id, args.strategy,
                               args.slowdown, args.execute, args.verbose, args.safety)
        

    return 1
