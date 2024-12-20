## isalnum()
use: Python string isalnum() function returns `True` if it’s made of alphanumeric characters only. A character is alphanumeric if it’s either an alpha or a number. If the string is empty, then isalnum() returns `False`.

```Python
s = 'HelloWorld2019'
print(s.isalnum())
```
Output: `True`

```Python
s = 'Hello World 2019'
print(s.isalnum())
```
Output: `False` as `s` has white spaces.

```Python
s = ''
print(s.isalnum())
```
Output: `False` as it is an empty string


