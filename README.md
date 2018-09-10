# rapid-transport

`rapid-transport` is the accompanying source code of the paper
"Critically fast pick-and-place with suction cups" submitted to ICRA
2019. `rapid-transport` implements a motion planning pipeline that
produces fast and robust motion for object transportation.

A video of the experiments of this pipeline can be found below.
[![nil](http://img.youtube.com/vi/b9H-zOYWLbY/0.jpg)](http://www.youtube.com/watch?v=b9H-zOYWLbY "rapid-transport demo")


You are welcome to clone this package and experiment with the code. If
you have any questions pertaining to the code, do raise an issue via
the Github Issue Tracking system.  See below for some instructions on
installing and running.

## Installation

`rapid-transport` has two main dependencies:
[`toppra`](https://github.com/hungpham2511/toppra) and
[`OpenRAVE`](https://github.com/rdiankov/openrave).

Here are instructions on installing both packages for your reference:
``` bash
# install openrave
git clone https://github.com/crigroup/openrave-installation
cd openrave-installation
./install-dependencies.sh
./install-osg.sh
./install-fcl.sh
./install-openrave.sh

# install toppra
git clone https://github.com/hungpham2511/toppra && cd toppra/
pip install -r requirements.txt --user
python setup.py install --user
```

Now you can run `rapid-transport`
```bash
git clone https://github.com/hungpham2511/toppra-object-transport && cd toppra-object-transport
pip install -r requirements.txt --user
python setup.py install --user
```
For more information on toppra, see [this page](https://hungpham2511.github.io/toppra/).

## Experiment with the code

`rapid-transport` provides two executables that you might be
interested in: i) `generate_contact_constraint.py` and ii)
`transport.paper pick-demo`.

### Running `generate_contact_constraint.py`

``` bash
cd <rapid-transport>dir
python scripts/generate_contact_constraint.py
```

This executable performs the contact stability constraint
approximation procedure described in the paper, and outputs the
constraint as a numpy archive.

If you are just trying out, check `data/contacts.yaml` and
`data/contact_data/analytical_rigid1234.npyz`. This is a pre-generated
constraint that is used in the paper.

### Running `pick-demo`

``` bash
transport.paper pick-demo --scene "scenarios/exp2.scenario.yaml"
```

This executable runs the demo discussed in the paper. You will need
OpenRAVE to run this one.



## Citation


