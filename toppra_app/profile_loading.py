import yaml, os


DEFAULT_CONF_FILE = "~/git/hung/Papers/2018-TOPP-RA-application/toppra_application_code/data/conf.yaml"


class Database(object):
    """ A simple database to retrieve and insert contact, object, robot, trajectory and algorithm
    profiles.

    """
    def __init__(self, conf_file=DEFAULT_CONF_FILE):
        with open(os.path.expanduser(conf_file)) as f:
            self.options = yaml.load(f.read())
        self.options['contact_profile_dir'] = os.path.expanduser(self.options['contact_profile_dir'])
        self.options['object_profile_dir'] = os.path.expanduser(self.options['object_profile_dir'])
        self.options['robot_profile_dir'] = os.path.expanduser(self.options['robot_profile_dir'])
        self.options['trajectory_profile_dir'] = os.path.expanduser(self.options['trajectory_profile_dir'])
        self.options['algorithm_profile_dir'] = os.path.expanduser(self.options['algorithm_profile_dir'])
        self.options['contact_data_dir'] = os.path.expanduser(self.options['contact_data_dir'])

        with open(self.options['contact_profile_dir']) as f:
            self.all_contacts = yaml.load(f.read())
        with open(self.options['object_profile_dir']) as f:
            self.all_objects = yaml.load(f.read())
        with open(self.options['trajectory_profile_dir']) as f:
            self.all_trajectories = yaml.load(f.read())
        with open(self.options['robot_profile_dir']) as f:
            self.all_robots = yaml.load(f.read())
        with open(self.options['algorithm_profile_dir']) as f:
            self.all_algorithms = yaml.load(f.read())

    def retrieve_profile(self, obj_id, table):
        """ Retrieve an object profile.

        Parameters
        ----------
        obj_id: str
            Object profile identifier.
        table
            Kind of profile. Can be "contact", "robot", ...etc

        Returns
        -------
        out: dict
            Object profile.
        """
        if table == "contact":
            selected_table = self.all_contacts
        elif table == "object":
            selected_table = self.all_objects
        elif table == "robot":
            selected_table = self.all_robots
        elif table == "trajectory":
            selected_table = self.all_trajectories
        elif table == "algorithm":
            selected_table = self.all_algorithms
        else:
            raise IOError, "Table [{:}] not found!".format(table)

        if obj_id not in selected_table:
            raise ValueError, "Object with id [{:}] not found in table [{:}]".format(obj_id, table)
        return selected_table[obj_id]

    def insert_profile(self, obj_profile, table):
        """ Insert a new profile.

        Parameters
        ----------
        obj_profile: dict
            The object profile to be inserted.
        table: str
            Name of the table.

        """
        if table == 'contact':
            self.all_contacts[obj_profile['id']] = obj_profile
            with open(self.options['contact_profile_dir'], 'w') as yaml_file:
                yaml_file.write(yaml.dump(self.all_contacts))
        elif table == "trajectory":
            self.all_trajectories[obj_profile['id']] = obj_profile
            with open(self.options['trajectory_profile_dir'], 'w') as yaml_file:
                yaml_file.write(yaml.dump(self.all_trajectories))
        else:
            raise NotImplementedError, "Insertion for table {:} is not implemented!".format(table)

    def get_contact_data_dir(self):
        """ Return the path to data storage directory.

        Returns
        -------
        out: str
            Data directory.
        """
        return self.options['contact_data_dir']

    def get_trajectory_data_dir(self):
        """ Return the path to data storage directory.

        Returns
        -------
        out: str
            Data directory.
        """
        return self.options['contact_data_dir']

