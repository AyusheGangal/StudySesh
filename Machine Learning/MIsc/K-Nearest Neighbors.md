The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

### The KNN Algorithm
1.  Load the data
2.  Initialize K to your chosen number of neighbors
3. For each example in the data
	1. Calculate the distance between the query example and the current example from the data.
	2. Add the distance and the index of the example to an ordered collection
	
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels
8. If classification, return the mode of the K labels

![[Screenshot 2022-11-15 at 2.00.35 PM.png]]

