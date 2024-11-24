import notebook.app
import sys

# Remove '--no-mathjax' from the arguments list
sys.argv = [arg for arg in sys.argv if arg != '--no-mathjax']

notebook.app.main()
