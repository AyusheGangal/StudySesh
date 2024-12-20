### Manual Implementation
```Python
import sys

class MaxHeap:

def __init__(self, maxsize):
	self.maxsize = maxsize
	self.size = 0
	self.Heap = [0] * (self.maxsize + 1)
	self.Heap[0] = sys.maxsize
	self.FRONT = 1

def parent(self, pos):
	return pos // 2

def leftChild(self, pos):
	return 2 * pos

def rightChild(self, pos):
	return (2 * pos) + 1

def isLeaf(self, pos):
	if pos >= (self.size//2) and pos <= self.size:
		return True
	return False

def swap(self, fpos, spos):
	self.Heap[fpos], self.Heap[spos] = (self.Heap[spos], self.Heap[fpos])

def maxHeapify(self, pos):
if not self.isLeaf(pos):
	if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or self.Heap[pos] < self.Heap[self.rightChild(pos)]):
		if (self.Heap[self.leftChild(pos)] > self.Heap[self.rightChild(pos)]):
			self.swap(pos, self.leftChild(pos))
			self.maxHeapify(self.leftChild(pos))
		else:
			self.swap(pos, self.rightChild(pos))
			self.maxHeapify(self.rightChild(pos))


def insertNode(self, element):
	if self.size >= self.maxsize:
		return
	self.size += 1
	self.Heap[self.size] = element
	current = self.size

	while (self.Heap[current] > self.Heap[self.parent(current)]):
		self.swap(current, self.parent(current))
		current = self.parent(current)

def Print(self):
	for i in range(1, (self.size // 2) + 1):
		print("PARENT NODE : " + str(self.Heap[i]) + " LEFT CHILD : " + str(self.Heap[2 * i]) + " RIGHT CHILD : " + str(self.Heap[2 * i + 1]))

  
def extractMaximum(self):
	popped = self.Heap[self.FRONT]
	self.Heap[self.FRONT] = self.Heap[self.size]
	self.size -= 1
	self.maxHeapify(self.FRONT)
	return popped
```

### Python Implementation using Built-in Functions
Heapq class will be used for implementing the built-in Max Heap in Python. And in this class, by default Min Heap is implemented. For Implementing Max Heap, we have to multiply all the keys with -1.

```Python
from heapq import heappop, heappush, heapify

heap = []
heapify(heap)

heappush(heap, -1 * 5)
heappush(heap, -1 * 9)
heappush(heap, -1 * 1)
heappush(heap, -1 * 11)
heappush(heap, -1 * 28)
heappush(heap, -1 * 19)
heappush(heap, -1 * 7)
heappush(heap, -1 * 2)
heappush(heap, -1 * 8)

print("The Head value of heap is : "+str(-1 * heap[0]))

print("Heap elements are : ")
for i in heap:
    print(-1 * i, end = ' ')
print("\n")

element = heappop(heap)
```

### Built-in Functions
![[Screenshot 2023-03-03 at 2.12.36 PM.png]]

### Heaps to Implement Heap Sort
A heapsort can be implemented by pushing all values onto a heap and then popping off the smallest values one at a time
```Python
def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
```
Output: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

This is similar to `sorted(iterable)`, but unlike `sorted()`, this implementation is not stable.
Heap elements can be tuples. This is useful for assigning comparison values (such as task priorities) alongside the main record being tracked:
```Python
h = []
heappush(h, (5, 'write code'))
heappush(h, (7, 'release product'))
heappush(h, (1, 'write spec'))
heappush(h, (3, 'create tests'))
heappop(h)
```
Output: `(1, 'write spec')`

### Heaps to Implement Priority Queue
Heaps are commonly used to implement [[Priority Queues]]. It presents several implementation challenges though:
- Sort stability: how do you get two tasks with equal priorities to be returned in the order they were originally added?
	- **solution:** A solution to the first two challenges is to store entries as 3-element list including the priority, an entry count, and the task. The entry count serves as a tie-breaker so that two tasks with the same priority are returned in the order they were added. And since no two entry counts are the same, the tuple comparison will never attempt to directly compare two tasks.****
- Tuple comparison breaks for (priority, task) pairs if the priorities are equal and the tasks do not have a default comparison order.
- If the priority of a task changes, how do you move it to a new position in the heap?
- Or if a pending task needs to be deleted, how do you find it and remove it from the queue?

Another solution to the problem of non-comparable tasks is to create a wrapper class that ignores the task item and only compares the priority field:
```Python
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
```

The remaining challenges revolve around finding a pending task and making changes to its priority or removing it entirely. Finding a task can be done with a dictionary pointing to an entry in the queue.

Removing the entry or changing its priority is more difficult because it would break the heap structure invariants. So, a possible solution is to mark the entry as removed and add a new entry with the revised priority:
```Python
pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')
```

