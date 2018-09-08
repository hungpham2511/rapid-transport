# rapid-transport

`rapid-transport` is the accompanying source code of the paper
"Critically fast pick-and-place with suction cups" submitted to ICRA
2019. `rapid-transport` implements a motion planning pipeline that
produces fast and robust motion for object transportation.

A video of the experiments of this pipeline can be found below.
[![nil](http://img.youtube.com/vi/b9H-zOYWLbY/0.jpg)](http://www.youtube.com/watch?v=b9H-zOYWLbY "rapid-transport demo")


You are welcome to clone this package and experiment with the code. If
you have any questions pertaining to the code, do go ahead and raise
an issue via Github Issue Tracking system. See below for some
instructions on how to install the package and play with the code.

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

## Trying the code

## Citation

