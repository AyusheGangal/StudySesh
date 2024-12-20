At the core of the Factory Method pattern is the concept of delegation. Instead of the client code directly creating objects, it delegates the responsibility to a Factory Method.

This method resides in an abstract Creator class or interface, defining an interface for creating objects. Concrete subclasses of the Creator implement this Factory Method, allowing them to create specific instances of objects. This delegation promotes loose coupling between the client code and the objects it uses, enhancing flexibility and maintainability.

#### Key Components of the Factory Method Pattern
Let’s break down the Factory Method pattern into its essential components.

1. **Creator:** The Creator is an abstract class or interface. It declares the Factory Method, which is essentially a method for creating objects. The Creator provides an interface for creating products but doesn’t specify their concrete classes.
2. **Concrete Creator:** Concrete Creators are the subclasses of the Creator. They implement the Factory Method, deciding which concrete Product class to instantiate. In other words, each Concrete Creator specializes in creating a particular type of product.
3. **Product:** The Product is another abstract class or interface. It defines the type of objects the Factory Method creates. These products share a common interface, but their concrete implementations can vary.
4. **Concrete Product:** Concrete Products are the subclasses of the Product. They provide the specific implementations of the products. Each Concrete Product corresponds to one type of object created by the Factory Method.

![[Screenshot 2024-01-19 at 3.49.27 PM.png]]

