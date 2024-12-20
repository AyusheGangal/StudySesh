
### <mark style="background: #D2B3FFA6;">Adding Features</mark>
For structured data problems, creating brand new training examples can be difficult but we can add features which can be useful.
Using an example of "Restaurant Recommendation", an app which recommends restaurant options to users. Assuming it is performing poorly by frequently recommending restaurants with poor or no vegetarian options to users which are clearly vegetarian from their order history. In this example, you can't synthesize new people as there are fixed people in the pool and there are so many (fixed number) of restaurants around. So here we can improve performance by adding features like:
1. Is the person vegetarian (based on past orders)?
2. Does the restaurant have vegetarian options? (based on menu)

<mark style="background: #ADCCFFA6;">Product Recommendations</mark>
- Over the last couple of years, there has been a shift from Collaborative filtering to Content-based filtering approaches. 
- Collaborative filtering is loosely an approach that looks at the user, tries to figure out who is similar to that user and then recommends things to the user that people like them also liked. 
- In contrast, a content based filtering approach will tend to look at the user as a person and look at the description of the restaurant or look at the menu of the restaurants and look at other information about the restaurant, to see if that restaurant is a good match for the user or not. 
- The advantage of content based filtering is that even if there's a new restaurant or a new product that hardly anyone else has liked by actually looking at the description of the restaurant, rather than just looking at who else like the restaurants, it can more quickly make good recommendations. This is sometimes also called the <mark style="background: #FFF3A3A6;">Cold Start Problem</mark>. 

### <mark style="background: #D2B3FFA6;">Data Iteration for Structured Data</mark>
![[Screenshot 2023-01-14 at 10.08.55 PM.png|300]]
- Error analysis can be harder if there is no good baseline (such as HLP) to compare to.
- Error analysis, user feedback and benchmarking to competitors can all provide inspiration for features to add.



