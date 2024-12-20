A linked list is a sequential list of nodes that hold data which point to other nodes also containing data. The last node always has a null reference.![[Screenshot 2023-03-01 at 8.47.31 PM.png]]

#### Uses of Linked Lists
![[Screenshot 2023-03-01 at 8.48.31 PM.png|400]]

#### Terminology
![[Screenshot 2023-03-01 at 8.50.08 PM.png|400]]

#### Singly vs Doubly Linked Lists
![[Screenshot 2023-03-01 at 9.56.50 PM.png|400]]

##### Pros and Cons on Singly and Doubly Linked Lists
	![[Screenshot 2023-03-01 at 9.58.14 PM.png|400]]
- Singly LL uses less memory because one less pointer and pointers can take up to a lot of memory.

##### Insertion in Singly Linked List

**1. Create a Class For A Node . Initialize the Properties which are needed in a Node . (value and link):**
```Python
class Node:
	def __init__(self,value,link = None):  
	self.value = value  
	self.link = link
```

**2. Create a Class For your linked list which will have an initialized First Node as None and other needed methods.**
```Python
class LinkedList:  
	def __init__(self):  
	self.firstNode= None
```

**3. Create method for Inserting at beginning**: Create a New Node. Then check if there is already a node in your list then link your new node to first node and then assign first node into new node. And if there is no nodes then assign first node as new node .
```Python
def insertAtbeginning(self,value):  
	newNode = Node(value)  
	if self.firstNode != None:  
		newNode.link = self.firstNode  
		self.firstNode = newNode  
	else:  
		self.firstNode = newNode
```

**4. Create method for Inserting at end**: Create a new node and if there is already nodes in linked list then travel through the nodes.
```Python
def insertAtEnd(self,value):  
	newNode = Node(value)  
	if self.firstNode != None:  
		current = self.firstNode  
		while current.link != None:  
			current = current.link  
		current.link = newNode  
	else:  
		self.firstNode = newNode
```

**5. Create a method which can display all the nodes of your linked list**
```Python
def display(self):  
	if self.firstNode == None:  
		print("List is empty")  
		return  
	current = self.firstNode  
	while current.link != None:  
		print(current.value)  
		current = current.link
```

**6. Create a method which can delete the element by its value , and check if the list is empty then return it and if its not empty then check for the first element that, is it the target ? if yes then return . And if first node is not the target then travel the list and find .**