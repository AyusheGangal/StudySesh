1. A **priority queue** is an abstract data-type similar to a regular [queue](https://en.wikipedia.org/wiki/Queue_(abstract_data_type) "Queue (abstract data type)") or [stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type) "Stack (abstract data type)") data structure. Each element in a priority queue has an associated _priority._ 
2. In a priority queue, elements with high priority are served before elements with low priority. In some implementations, if two elements have the same priority, they are served in the same order that they were enqueued in. In other implementations, the order of elements with the same priority is undefined.

While priority queues are often implemented using [heaps](https://en.wikipedia.org/wiki/Heap_(data_structure) "Heap (data structure)"), they are conceptually distinct from heaps. A priority queue is an abstract data structure like a [list](https://en.wikipedia.org/wiki/List_(abstract_data_type) "List (abstract data type)") or a [map](https://en.wikipedia.org/wiki/Associative_array "Associative array"); just as a list can be implemented with a [linked list](https://en.wikipedia.org/wiki/Linked_list "Linked list") or with an [array](https://en.wikipedia.org/wiki/Array_data_structure "Array data structure"), a priority queue can be implemented with a heap or another method such as an unordered array.

![[Screenshot 2023-03-03 at 2.34.39 PM.png]]

## Naive Implementation
There are a variety of simple, usually inefficient, ways to implement a priority queue. They provide an analogy to help one understand what a priority queue is.
For instance, one can keep all the elements in an unsorted list (_O_(1) insertion time). Whenever the highest-priority element is requested, search through all elements for the one with the highest priority. (_O_(_n_) pull time)
```Python
insert(node)
{
  list.append(node)
}
```

```Python
pull()
{
  highest = list.get_first_element()
  foreach node in list
  {
     if highest.priority < node.priority
     {
         highest = node
     }
  }
  list.remove(highest)
  return highest
}
```

In another case, one can keep all the elements in a priority sorted list (_O_(n) insertion sort time), whenever the highest-priority element is requested, the first one in the list can be returned. (_O_(1) pull time)
```Python
insert(node)
{
  foreach (index, element) in list
  {
    if node.priority < element.priority
    {
       list.insert_at_index(node,index)
       break
    }
  }
}
```

```Python
pull()
{
    highest = list.get_at_index(list.length-1)
    list.remove(highest)
    return highest
}
```

## Summary of running times
![[Screenshot 2023-03-03 at 2.41.06 PM.png]]

## Applications of Priority Queues
1. Bandwidth management: Priority queuing can be used to manage limited resources such as [bandwidth](https://en.wikipedia.org/wiki/Bandwidth_(computing) "Bandwidth (computing)") on a transmission line from a [network](https://en.wikipedia.org/wiki/Computer_network "Computer network") [router](https://en.wikipedia.org/wiki/Router_(computing) "Router (computing)"). In the event of outgoing [traffic](https://en.wikipedia.org/wiki/Traffic "Traffic") queuing due to insufficient bandwidth, all other queues can be halted to send the traffic from the highest priority queue upon arrival.
2. Discrete event simulation: The events are added to the queue with their simulation time used as the priority. The execution of the simulation proceeds by repeatedly pulling the top of the queue and executing the event thereon.
3. Dijkstra's algorithm
4. Huffman coding: Huffman coding requires one to repeatedly obtain the two lowest-frequency trees. A priority queue is one method of doing this.
5. Best-first search algorithms: [Best-first search](https://en.wikipedia.org/wiki/Best-first_search "Best-first search") algorithms, like the [A* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm "A* search algorithm"), find the shortest path between two [vertices](https://en.wikipedia.org/wiki/Vertex_(graph_theory) "Vertex (graph theory)") or [nodes](https://en.wikipedia.org/wiki/Node_(graph_theory) "Node (graph theory)") of a [weighted graph](https://en.wikipedia.org/wiki/Weighted_graph "Weighted graph"), trying out the most promising routes first. A priority queue (also known as the _fringe_) is used to keep track of unexplored routes; the one for which the estimate (a lower bound in the case of A*) of the total path length is smallest is given highest priority.
6. ROAM triangulation algorithm
7. Prim's algorithm for minimum spanning tree