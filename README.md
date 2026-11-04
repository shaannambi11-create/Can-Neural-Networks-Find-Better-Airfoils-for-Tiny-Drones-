# AI-guided wing design for tiny, slow drones

Code and results for my paper "Can Neural Networks Find Better Airfoils for Tiny Drones? A Computational Study at Low Reynolds Numbers".

I used the free program XFOIL to test about 11,400 wing shapes at a Reynolds number of 20,000. A small neural network guided one search, and I compared it with random search in 10 repeats. This is a computer study only. I did not build or test real wings.

## What you need
- Python
- XFOIL version 6.99 (not included here; get it from its author)
- pandas, NumPy, SciPy, matplotlib, scikit-learn

## Scripts (in the scripts folder)
make_wings.py, run_all_wings.py, train_ai.py, search_wings.py, ai_loop_filtered.py, repeat_search.py, extra_analysis.py, validate_xfoil.py, make_figures.py, count_wings.py, pick_wings.py

## See the results of the 10 repeats
python scripts\repeat_search.py summary

The AI-guided search did better than random search in 8 repeats, tied in 1, and did worse in 1. The standard NACA 2412 wing scores 3.95.

## Limits
- The quality filter limits are my own judgment, not from a standard.
- At other Reynolds numbers, most wings failed my filter.
- Most of the gain over the NACA 2412 comes from the thin, curved shape, not from the AI.

## Author
Shaan Gopal Nambi, International Indian School Riyadh

## License
MIT
