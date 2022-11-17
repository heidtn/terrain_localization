# TerLoc

This project attempts to expand upon the work done in the EZ-RASSOR project (https://github.com/FlaSpaceInst/EZ-RASSOR) for localization using elevation maps of a given terrain.  In the EZ-RASSOR project, this is called park ranger.

The general idea is if the rough position of an agent is known in an environment (say within 40 miles) and an elevation map is available of the region, can depth sensors from the rover be used to localize on elevation.

It is a collection of tests and experiments with the hope of furthering the general concept.


# Approach
- generate test terrain
- create robot "sim"