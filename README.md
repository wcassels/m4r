# M4R
Codebase for my 4th year research project in Mathematics, concerning the solution of PDEs (in particular the heat equation) using RBF methods.

Scripts:
- general_utils.py : Setup, and time-stepping functions for both the original and alternative solution procedures, on general domains
- rect_utils.py : As above but specialised to rectangular with uniform node arrangements (old, not as efficient)
- analyticals.py : Numerical approximations of the analytical solutions for the tests given in Sarler & Vertnik, with corrections
- node_utils.py : Generating, loading and saving node configurations
- normal_derivs.py : Evaluating inward normal derivatives for circular and rectangular boundaries

The remaining files are example scripts for generating results for specific tests/configurations (generally messy, with a lot of commented out code).
