import os
mydir = os.path.dirname(__file__)
initf = os.path.join(mydir, 'ML-modules', '__init__.py')
if not os.path.exists(initf):
    open(initf, 'wb').close()
