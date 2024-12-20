Design patterns exhibit diversity in complexity, detail, and applicability across a system’s scope. At the foundational level are idiomatic patterns, specific to a single programming language.

Ascending in scope, are the architectural patterns. Applicable across various languages, these patterns facilitate the design of an entire application’s architecture. 

Classification also occurs based on intent:
1. **Creational Patterns:** Facilitate flexible object creation and code reuse.
2. **Structural Patterns:** Assembling objects and classes into larger structures while retaining their adaptability and efficiency
3. **Behavioral Patterns:** Efficient interaction and allocation of responsibilities between objects, ensuring effective communication.
#### Creational Patterns
1. [[Singleton Pattern]]: _Ensures that a class has only one instance and provides a global point of access to that instance._
2. [[Factory Method Pattern]]: _Defines an interface for creating objects, allowing subclasses to decide which class to instantiate._
3. **Abstract Factory Pattern**: _Provides an interface for creating families of related or dependent objects without specifying their concrete classes._
4. **Builder Pattern**: _Separates the construction of a complex object from its representation, allowing the same construction process to create_ different representations.
5. **Prototype Pattern**: _Creates new objects by copying an existing object, avoiding the overhead of creating objects from scratch._

#### Structural Patterns
1. **Adapter Pattern**: _Converts the interface of a class into another interface that clients expect, enabling classes with incompatible interfaces to work together._
2. **Bridge Pattern:** _Decouples an abstraction from its implementation, allowing both to evolve independently._
3. **Composite Pattern**: _Composes objects into tree structures to represent part-whole hierarchies, making it easier to work with individual objects and compositions._
4. **Decorator Pattern**: _Dynamically adds responsibilities to objects, providing a flexible alternative to subclassing for extending functionality._
5. **Facade Pattern**: _Provides a simplified interface to a complex subsystem, making it easier to use and understand._
6. **Flyweight Pattern:** _Shares instances of objects to support large numbers of fine-grained objects efficiently._
7. **Proxy Pattern**: _provide a substitute or placeholder for another object to control access to the original object._

#### Behavioral Patterns
1. **Chain of Responsibility Pattern:** _Creates a chain of objects that can handle requests, avoiding coupling the sender with its receivers._
2. **Command Pattern**: _Turns a request into a stand-alone object, allowing parameterization of clients with different requests._
3. **Interpreter Pattern:** _Defines a grammar for a language and an interpreter to interpret sentences in the language._
4. **Iterator Pattern**: _Provides a way to access elements of a collection without exposing its underlying representation._
5. **Mediator Pattern**: _Defines an object that centralizes communication between multiple objects, reducing direct dependencies between them._
6. **Memento Pattern**: _Captures and restores an object’s internal state, allowing it to be restored to a previous state._
7. **Observer Pattern:** _Defines a dependency between objects, ensuring that when one object changes state, all its dependents are notified and updated automatically._
8. **State Pattern**:_Allows an object to change its behavior when its internal state changes, enabling cleaner, more maintainable conditional logic._
9. **Strategy Pattern:** _Defines a family of algorithms, encapsulates each one and makes them interchangeable. Clients can choose an algorithm from this family without modifying their code._
10. **Template Method Pattern:** _Defines the structure of an algorithm in a superclass but lets subclasses override specific steps of the algorithm._
11. **Visitor Pattern:** _Separates an algorithm from an object structure, allowing new operations to be added without modifying the objects themselves._

