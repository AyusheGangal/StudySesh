- The Singleton pattern falls under the [creational design patterns](https://en.wikipedia.org/wiki/Creational_pattern) category. 
- It is particularly useful when you want to <mark style="background: #ADCCFFA6;">ensure that a class has only one instance throughout the application’s lifecycle</mark>. 
- This can be <mark style="background: #D2B3FFA6;">valuable in scenarios where you need a single point of control, such as managing configurations, database connections, or logging services</mark>.

>In mathematics, singleton is defined as “a set which contains exactly one element”

In essence, <mark style="background: #ADCCFFA6;">a Singleton pattern restricts the instantiation of a class to a single object. This means that regardless of how many times the code requests an instance of the class, it will always receive the same instance.</mark> 

Think of it as a gatekeeper for ensuring that a particular class remains unique and unified throughout an application.

The Singleton pattern’s <mark style="background: #D2B3FFA6;">primary goal is to provide a single point of access to that single instance, simplifying the process of managing shared resources, centralizing control, and promoting consistency in behavior.</mark> This uniqueness can prove to be incredibly valuable when dealing with resources that need to be shared across different parts of the codebase.

![[Screenshot 2024-01-19 at 2.20.14 PM.png]]

#### Benefits of Using the Singleton Pattern
Can one pattern really simplify resource management, enhance global access, and streamline code? Here is the hidden potential of Singleton and its benefits.

1. **Single Instance**: Ensures that only one instance of the class is created, preventing multiple instances from causing conflicts or consuming unnecessary resources.
2. **Global Access**: Provides a global point of access to the instance, making it easy to share data or functionality across different parts of the application.
3. **Resource Management**: Helps manage resources that should be shared, such as database connections, without creating multiple connections and overwhelming the system.
4. **Lazy Initialization**: Allows for efficient resource usage by creating the instance only when it is actually needed.

#### Implementing the Singleton Pattern in Python
In Python, the Singleton Pattern is often realized differently from the [Gang of Four’s](https://www.goodreads.com/en/book/show/85009) original conception.
<mark style="background: #ADCCFFA6;">The original design of the Singleton class forbids normal instantiation and instead offers a static class method that returns the singleton instance.</mark> 

Unlike the rigid Singleton class, <mark style="background: #D2B3FFA6;">Python enables normal instantiation with a custom `__new__` method for obtaining the singleton instance</mark>.

###### Basic Class Implementation
```Python
class Singleton:  
	_instance = None  
  
	def __new__(cls):  
		if cls._instance is None:  
			cls._instance = super().__new__(cls)  
		return cls._instance
```

In this example, the `__new__` method is overridden to ensure that only one instance of the class is created. If the instance does not exist, a new one is created; otherwise, the existing instance is returned.

###### Decorator Implementation
In this implementation, the `singleton` decorator function manages instances of decorated classes. It ensures that only one instance of each class exists and returns the same instance when the class is instantiated. The `SingletonClass` is decorated with `@singleton`, turning it into a singleton, and any attempts to create new instances will return the existing instance.

```Python
def singleton(cls):  
	instances = {} # Dictionary to store instances of different classes  
  
	def get_instance(*args, **kwargs):  
		# If class instance doesn't exist in the dictionary  
		if cls not in instances:  
			# Create a new instance and store it  
			instances[cls] = cls(*args, **kwargs)  
		return instances[cls] # Return the existing instance  
  
	# Return the closure function for class instantiation  
	return get_instance  
  
  
@singleton # Applying the singleton decorator  
class SingletonClass:  
	def __init__(self, data):  
		self.data = data  
  
	def display(self):  
		print(f"Singleton instance with data: {self.data}")  
  
  
# Creating instances of SingletonClass using the decorator  
instance1 = SingletonClass("Instance 1")  
instance2 = SingletonClass("Instance 2")  
  
# Both instances will refer to the same instance  
instance1.display() # Output: Singleton instance with data: Instance 1  
instance2.display() # Output: Singleton instance with data: Instance 1
```

>The most common use case for applying Singleton design pattern is when dealing with databases.

#### Singleton Pattern for Threading
While the decorator implementation works in most cases, it might not be thread-safe, especially in a multi-threaded environment. To achieve thread safety and lazy initialization, consider using the following approach:

```Python
import threading  
  
class ThreadSafeSingleton:  
	_instance = None  
	_lock = threading.Lock()  
  
	def __new__(cls):  
		with cls._lock:  
			if cls._instance is None:  
				cls._instance = super().__new__(cls)  
		return cls._instance
```

Here, a threading lock is used to ensure that only one thread can create the instance at a time, preventing race conditions. <mark style="background: #D2B3FFA6;">Leveraging lazy initialization means that the class instance is created only upon the initial object creation.</mark>

#### Potential Drawbacks and Considerations
While the Singleton pattern provides clear advantages, it’s imperative to recognize its potential limitations. In fact, the Singleton pattern, despite its solutions, is occasionally labeled an “anti-pattern” due to the following reasons:
1. [**Single Responsibility**](https://en.wikipedia.org/wiki/Single-responsibility_principle) **Violation:** _Simultaneously addressing two concerns, the Singleton pattern can blur responsibilities._
2. **Global Coupling:** _The globally accessible Singleton instance can foster tight interdependence between application sections, potentially complicating maintenance and testing._
3. **Testing Dilemmas:** _Testing components reliant on a Singleton can pose difficulties, given the influence of the global state on test outcomes._
4. **Multithreading Complexities:** _In multithreaded environments, special precautions are necessary to avert multiple thread-based Singleton creation._

#### Real-world Use Cases: Singleton in Action
The Singleton pattern is commonly used in various real-world scenarios:
1. **Database Connection Pools:** _Enhancing database interaction efficiency via a unified connection pool._
2. **Logger Services:** _Centralizing application logging through a single logger instance._
3. **Configuration Management:** _Ensuring a solitary configuration manager instance oversees application settings._
4. **Hardware Access:** _Controlling access to hardware resources, such as a printer or sensor, through a single instance._