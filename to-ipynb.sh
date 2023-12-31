#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: to-ipynb.sh <py file>"
    echo "Converts py file to ipynb."
    exit 1
fi
pyfile=$1

echo "Converting $pyfile to "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"..."

sed 's/#pip install/pip install/' $pyfile | sed '/plt.ion/d' > temp.py
ipynb-py-convert temp.py "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"
rm temp.py

