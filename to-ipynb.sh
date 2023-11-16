#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: generate-ipynb.sh <py file>"
    echo "Converts py file to ipynb."
    exit 1
fi
pyfile=$1

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/#pip install/pip install/' $pyfile
else
	sed -i 's/#pip install/pip install/' $pyfile
fi

echo "Converting $pyfile to "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"..."
ipynb-py-convert $pyfile "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"

if [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' 's/pip install/#pip install/' $pyfile
else
	sed -i '' 's/pip install/#pip install/' $pyfile
fi


