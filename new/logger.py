import logging

LOGGER = logging.getLogger()
fh = logging.FileHandler(filename='test.log')
formatter = logging.Formatter('%(name)s:%(levelname)s - %(message)s')
LOGGER.addHandler(fh)
LOGGER.setLevel(logging.DEBUG)



