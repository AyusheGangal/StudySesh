The [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "pickle: Convert Python objects to streams of bytes and back.") module implements binary protocols for serializing and de-serializing a Python object structure. _“Pickling”_ is the process whereby a Python object hierarchy is converted into a byte stream, and _“unpickling”_ is the inverse operation, whereby a byte stream (from a [binary file](https://docs.python.org/3/glossary.html#term-binary-file) or [bytes-like object](https://docs.python.org/3/glossary.html#term-bytes-like-object)) is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [[1]](https://docs.python.org/3/library/pickle.html#id7) or “flattening”; however, to avoid confusion, the terms used here are “pickling” and “unpickling”.

### Comparison with `json`[](https://docs.python.org/3/library/pickle.html#comparison-with-json "Link to this heading")
There are fundamental differences between the pickle protocols and [JSON (JavaScript Object Notation)](https://json.org/):
- JSON is a text serialization format (it outputs unicode text, although most of the time it is then encoded to `utf-8`), while pickle is a binary serialization format;
- JSON is human-readable, while pickle is not;
- JSON is interoperable and widely used outside of the Python ecosystem, while pickle is Python-specific;
- JSON, by default, can only represent a subset of the Python built-in types, and no custom classes; pickle can represent an extremely large number of Python types (many of them automatically, by clever usage of Python’s introspection facilities; complex cases can be tackled by implementing [specific object APIs](https://docs.python.org/3/library/pickle.html#pickle-inst));
- Unlike pickle, deserializing untrusted JSON does not in itself create an arbitrary code execution vulnerability.

## Data stream format[](https://docs.python.org/3/library/pickle.html#data-stream-format "Link to this heading")
The data format used by [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "pickle: Convert Python objects to streams of bytes and back.") is Python-specific. This has the advantage that there are no restrictions imposed by external standards such as JSON (which can’t represent pointer sharing); however it means that non-Python programs may not be able to reconstruct pickled Python objects.

By default, the [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "pickle: Convert Python objects to streams of bytes and back.") data format uses a relatively compact binary representation. If you need optimal size characteristics, you can efficiently [compress](https://docs.python.org/3/library/archiving.html) pickled data.

```Python
import pickle

# create dictionary containing all your data
data = {'stim': np.array([1, 2, 3]), 'response': np.array([6, 2, 0])}

# save data in pickle format
with open('my_data.pickle', 'wb') as f:
    pickle.dump(data, f)

# open data from file
with open('my_data.pickle', 'rb') as f:
    new_data_variable = pickle.load(f)

# now new_data_variable is equal to the dict:
# {'stim': np.array([1, 2, 3]), 'response': np.array([6, 2, 0])}```

