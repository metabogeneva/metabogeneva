# Metabogeneva

This package offers an implementation of several data analysis utilities developed at the [Biomedical and Metabolomic Analyses of the University of Geneva](https://farma-unites.unige.ch/en/rudaz-lab).
The package has been developed by Santiago Codesido Sánchez in collaboration with Julien Boccard and Víctor González Ruiz.

## Installation

```
pip install metabogeneva
```

Add the `notebook` extra (`metabogeneva[notebook]`) to use `IPythonMonitor` in Jupyter.

## Submodules

### BaRCA

The __barca__ submodule contains tools to build BaRCA (Bayesion Recursive Classification Analysis) models from multigroup datasets, and to classify new data according to these models.
For more information on its use, please read the relevant articles.

### IO

The __io__ submodule defines a common data format for dealing with metadata-enriched matrices in an efficient way.

## License

Metabogeneva is licensed under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
