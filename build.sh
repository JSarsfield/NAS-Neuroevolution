python setup.py build_ext
pip wheel .
var=$(find *.whl)
echo $var
pip install $var -U