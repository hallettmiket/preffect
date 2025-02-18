How to run Sphinx:

For a fresh install:
1. Install Sphinx and any libraries/templates you might need  (e.g. myst-parser, sphinx_rtd_theme)

2. In the folder of interest, run: 
> sphinx-quickstart 

- it will ask you to enter relevant information, e.g., authors, version number)
- for all other questions, go with the default option (just hit enter)

3. It will create a conf.py file. Edit it to indicate:
- extensions being used
- html themes
- what functions you want to exclude
- path to your folder of interest

4. Create a "rst" file for every program you want to create docs for:

Example: 
_utils.py module
======================

.. automodule:: _utils
   :members:
   :undoc-members:
   :show-inheritance:

5. Create the HTML file
> make html

6. Sphinx will report warnings and errors; they will mostly consist of issues with the docstrings that need to be resolved; fix them.

7. Run 'make html' again

8. Download and inspect the results. Make sure to download the entire HTML folder for the RTD formatting to appear.
- if you just download 1 file, it'll look like plain text HTML
