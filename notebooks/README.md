# Jupyter Notebooks

This folder contains auto-generated Jupyter notebooks from the lecture source files.

**⚠️ Do not edit these files directly!**

These notebooks are automatically generated during the CI/CD build process from the MyST Markdown source files in the `lectures/` directory. Any manual changes will be overwritten.

## How it works

1. Source files in `lectures/` are written in MyST Markdown
2. During the build process, `jupyter-book` executes the code and generates `.ipynb` files
3. The generated notebooks are copied to this folder
4. These notebooks are used by the Google Colab launch buttons on the website

## Opening in Google Colab

You can open any of these notebooks directly in Google Colab by:

1. Visiting the lecture website
2. Clicking the "Launch" button on any lecture page
3. Selecting Google Colab from the launcher options

Or construct the URL manually:
```
https://colab.research.google.com/github/QuantEcon/iuj_feb_2026/blob/main/notebooks/{notebook_name}.ipynb
```
