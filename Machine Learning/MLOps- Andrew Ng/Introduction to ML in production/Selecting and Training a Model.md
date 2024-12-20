##### <mark style="background: #D2B3FFA6;">Establish a baseline</mark> 
- Baseline helps to indicate what might be possible, in other words, what is the best that anyone could possibly hope for in terms of performance on a particular problem. In some cases (such as HLP) it also gives a sense of what is irreducible error/ Bayes error.
- Given a number of features in an ML system, Human Level Performance (HLP) is used to give us a point of comparison or a baseline that helps us decide where (on which feature) to focus our efforts on. 
	- Best practices for establishing a baseline are quite different, depending on whether you're working on unstructured or structured data. 
		- Unstructured data refers to data sets like images, maybe pictures of cats or audio, like our speech recognition example or natural language, like text from restaurant reviews. Unstructured data tends to be data that humans are very good at interpreting. In fact, humans evolve to be very good at understanding images and audio and maybe language as well. 
		- Structured data often comes in huge quantities in excel spreadsheets and it is difficult and often inefficient to use HLP as baseline for the predictions using structured data.
- Literature search for state-of-the-art/ open source
- Quick-and-dirty implementation
- Performance of older system

##### <mark style="background: #D2B3FFA6;">Why lower average error isn't good enough?</mark>
1. A machine learning system may have low average test set error, but if its performance on a set of disproportionately important examples isn't good enough, then the machine learning system will still not be acceptable for production deployment. 
	- Considering an example of Web Search, we have two types of queries:
		- <mark style="background: #ABF7F7A6;">Informational and Transactional queries:</mark> More generalized queries  like "best apple pie", "wireless plans", "latest movies" etc,. A web search engine wants to return the most relevant results, but users are willing to forgive maybe ranking the best result, Number two or Number three.
		- <mark style="background: #ABF7F7A6;">Navigational queries:</mark>  More specific queries like "Stanford", "Youtube", "Reddit" etc,. When a user has a very clear navigational intent, they will tend to be very unforgiving if a web search engine does anything other than that as the Number one ranked results and the search engine that doesn't give the right results will quickly lose the trust of its users.
	- Navigational queries in this context are a disproportionately important set of examples and if you have a learning algorithm that improves your average test set accuracy for web search but messes up just a small handful of navigational queries, that may not be acceptable for deployment. 
	- The challenge, of course, is that average test set accuracy tends to weight all examples equally, whereas, in web search, some queries are disproportionately important. Now one thing you could do is try to give these examples a higher weight.
	
2. Performance on key slices of the dataset.
	- Example # 1 - ML for loan approval: Make sure not to discriminate by ethnicity, gender, location, language or other projected attributes.
	- Example # 2 - Product recommendations from retailers: Be careful to treat all major users, retailers and product categories.
	
3. Rare classes- Skewed data distribution


